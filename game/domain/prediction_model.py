import pandas as pd
import numpy as np
from functools import reduce
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X, Z
from qulacs.gate import DenseMatrix
from qulacs import ParametricQuantumCircuit
from qulacs import Observable
from scipy.optimize import minimize
from dataclasses import dataclass

TimeMs = float


@dataclass
class TypeTime:
    word: str
    time_ms: TimeMs


class TypeTimePredictionModel:
    def __init__(self):
        pass

    def train(self, typetimes: list[TypeTime]):
        pass

    def partial_train(self, typetimes: list[TypeTime]):
        pass

    def reset(self):
        pass

    def predict_times(self, words: list[str]) -> list[TimeMs]:
        return [i * 2 for i in enumerate(words)]


class QuantumTypeTimePredictionModel():
    """
    コンストラクタ
    """
    def __init__(self, logs):
        self.logs = logs
        self._parsed_logs = self.parse_logs(logs)
        self._unique_words = pd.unique(self._parsed_logs["word"])

        self.nqubit = 5 ## qubitの数
        self.c_depth = 5 ## circuitの深さ
        self.time_step = 0.79  ## ランダムハミルトニアンによる時間発展の経過時間
        self.max_val = 3636363636
        self.min_val = 0.0
    
    def parse_logs(self, rawlogs):
      new_logs = []
      for log in self.logs:
          new_log = {}
          if log["type"] == "gameStart":
              new_log["type"] = log["type"]
              start_time = log["timestamp"]
              new_log["timestamp"] = start_time
              new_log["word"] = ""
              new_log["key"] = ""
              new_log["is_correct"] = True

          elif log["type"] == "keyPress":
              new_log["type"] = log["type"]
              new_log["timestamp"] = log["timestamp"] - start_time
              new_log["word"] = log["data"]["wordToType"]
              new_log["key"] = log["data"]["keyPressed"]
              if log["data"]["isCorrect"]:
                  new_log["is_correct"] = True
              else:
                  new_log["is_correct"] = False

          new_logs.append(new_log)
      return pd.DataFrame(new_logs)

    """
    logデータの解析を行う関数
    """
    def analyze_logs(self):
        self._parsed_logs = self.parse_logs(self.logs)
        word_history = {}
        for word in self._unqiue_words:
            word_history[word] = self._parsed_logs[self._parsed_logs["word"] == word]

        times = []
        for word in self._unqiue_words:
            time_to_type =  word_history[word]["timestamp"].max() - word_history[word]["timestamp"].min()
            times.append(time_to_type)

        return times

    """
    1文字を入力するとその文字キーの番号とそのキーの中での位置を返す関数
    """
    def value_from_arrangement_2(self,c):
        key_set = ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]

        string = ""
        for i in range(3):
            if c in key_set[i]:
                string += str(i+1)
                break
        for j in range(10):
            if key_set[i][j] == c:
                string += str(j)
        return string

    """
    配置をもとに文字列を数字にエンコードする関数
    """
    def my_charactor_encoder(self, words):
        res = []
        for word in words:
            string = ""
            for c in word:
                string += self.value_from_arrangement_2(c)
            res.append(int(string))
        return res

    """
    文字データの設定
    """
    def encode_word_data(self):
        xdata = self.my_charactor_encoder(self._unique_words)
        xdata_orig = xdata

        # 最小値，最大値の設定 0は空白文字，3636363636は5文字の最大値であるmmmmmのエンコード値
        min_val = 0.0
        max_val = 3636363636

        # 先頭に0を追加
        xdata.insert(0,0)

        # [-1,1]の範囲に正規化
        xdata = [(2*(x - min_val)/(max_val - min_val)) - 1 for x in xdata]

        return [xdata, xdata_orig]

    def encode_time_data(self):
        times = self.analyze_logs()
        times.insert(0,0)
        max_time = np.max(times)
        min_time = np.min(times)

        # [-1,1]の範囲に正規化
        times = [ 2*(time - min_time)/(max_time - min_time) - 1 for time in times]
        return [times, max_time, min_time]

    # xをエンコードするゲートを作成する関数
    def U_in(self, x):
        U = QuantumCircuit(self.nqubit)

        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)

        for i in range(self.nqubit):
            U.add_RY_gate(i, angle_y)
            U.add_RZ_gate(i, angle_z)

        return U

    ## fullsizeのgateをつくる関数.
    def make_fullgate(self, list_SiteAndOperator, nqubit):
        '''
        list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
        関係ないqubitにIdentityを挿入して
        I(0) * ... * _0(i_0) * ... * O_1(i_1) ...
        という(2**nqubit, 2**nqubit)行列をつくる.
        '''
        list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
        list_SingleGates = [] ## 1-qubit gateを並べてnp.kronでreduceする
        cnt = 0
        for i in range(nqubit):
            if (i in list_Site):
                list_SingleGates.append( list_SiteAndOperator[cnt][1] )
                cnt += 1
            else: ## 何もないsiteはidentity
                list_SingleGates.append(self.I_mat)

        return reduce(np.kron, list_SingleGates)

    # パラメータthetaを更新する関数
    def set_U_out(self, theta):
        parameter_count = self.U_out.get_parameter_count()

        for i in range(parameter_count):
            self.U_out.set_parameter(i, theta[i])

    # 入力x_iからモデルの予測値y(x_i, theta)を返す関数
    def qcl_pred(self, x, nqubit, obs,  U_out):
        state = QuantumState(nqubit)
        state.set_zero_state()

        # 入力状態計算
        self.U_in(x).update_quantum_state(state)

        # 出力状態計算
        U_out.update_quantum_state(state)

        # モデルの出力
        res = obs.get_expectation_value(state)

        return res

    # cost function Lを計算
    def cost_func(self, theta):
        '''
        theta: 長さc_depth * nqubit * 3のndarray
        '''
        # U_outのパラメータthetaを更新
        global U_out
        self.set_U_out(theta)

        # num_x_train個のデータについて計算
        y_pred = [self.qcl_pred(x, self.nqubit, self.obs, self.U_out) for x in self.x_train]

        # quadratic loss
        L = ((y_pred - self.y_train)**2).mean()

        return L

    """
    正規化された時間を復元する関数
    """

    def pred_time_trans(self, time, max_time, min_time):
        return (time + 1) * (max_time - min_time) / 2 + min_time

    def train(self):
        """
        パラメタの設定
        """
        self.nqubit = 5 ## qubitの数
        c_depth = 5 ## circuitの深さ
        time_step = 0.79  ## ランダムハミルトニアンによる時間発展の経過時間

        max_val = 3636363636
        min_val = 0.0

        #### 教師データを準備
        x_train, x_train_orig = self.encode_word_data()
        _y_train, max_raw_y, min_raw_y = self.encode_time_data()
        y_train = np.array(_y_train)
        max_time = max_raw_y
        min_time = min_raw_y

        self.x_train = x_train
        self.y_train = y_train

        state = QuantumState(self.nqubit) # 初期状態 |0>^n
        state.set_zero_state()

        ## 基本ゲート

        self.I_mat = np.eye(2, dtype=complex)
        self.X_mat = X(0).get_matrix()
        self.Z_mat = Z(0).get_matrix()

        #### ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
        ham = np.zeros((2**self.nqubit,2**self.nqubit), dtype = complex)
        for i in range(self.nqubit): ## i runs 0 to nqubit-1
            Jx = -1. + 2.*np.random.rand() ## -1~1の乱数
            ham += Jx * self.make_fullgate( [ [i, self.X_mat] ], self.nqubit)
            for j in range(i+1, self.nqubit):
                J_ij = -1. + 2.*np.random.rand()
                ham += J_ij * self.make_fullgate ([ [i, self.Z_mat], [j, self.Z_mat]], self.nqubit)

        ## 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
        diag, eigen_vecs = np.linalg.eigh(ham)
        time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj()) # e^-iHT

        # qulacsのゲートに変換しておく
        time_evol_gate = DenseMatrix([i for i in range(self.nqubit)], time_evol_op)

        # output用ゲートU_outの組み立て&パラメータ初期値の設定
        U_out = ParametricQuantumCircuit(self.nqubit)
        self.U_out = U_out
        for d in range(c_depth):
            U_out.add_gate(time_evol_gate)
            for i in range(self.nqubit):
                angle = 2.0 * np.pi * np.random.rand()
                U_out.add_parametric_RX_gate(i,angle)
                angle = 2.0 * np.pi * np.random.rand()
                U_out.add_parametric_RZ_gate(i,angle)
                angle = 2.0 * np.pi * np.random.rand()
                U_out.add_parametric_RX_gate(i,angle)

        # パラメータthetaの初期値のリストを取得しておく
        parameter_count = U_out.get_parameter_count()
        theta_init = [U_out.get_parameter(ind) for ind in range(parameter_count)]

        # オブザーバブルZ_0を作成
        obs = Observable(self.nqubit)
        obs.add_operator(2.,'Z 0') # オブザーバブル2 * Zを設定。ここで2を掛けているのは、最終的な<Z>の値域を広げるためである。未知の関数に対応するためには、この定数もパラメータの一つとして最適化する必要がある。
        self.obs = obs

        result = minimize(self.cost_func, theta_init, method='Nelder-Mead')

        # 最適化によるthetaの解
        theta_opt = result.x

        self.set_U_out(theta_opt)

    def predict_time(self, input_words: list[str]):
      pred_times = []
      for pred_word in input_words:
          pred_data = self.my_charactor_encoder([pred_word])
          pred_data = [(2*(x - self.min_val)/(self.max_val - self.min_val)) - 1 for x in pred_data]
          pred_time = self.pred_time_trans(self.qcl_pred(pred_data[0], self.nqubit, self.obs, U_out), self.time_span[1], self.time_span[0])
          # print("Predicted time: ", pred_time, pred_word)
          pred_times.append(pred_time)
      return pred_times

