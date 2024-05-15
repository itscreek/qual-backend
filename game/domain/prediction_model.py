from dataclasses import dataclass
import json
import pandas as pd
import numpy as np
from functools import reduce
import csv
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X, Z, DenseMatrix
from qulacs import ParametricQuantumCircuit, Observable
from scipy.optimize import minimize

TimeMs = float

@dataclass
class TypeTime:
    word: str
    time_ms: TimeMs

class QuantumTypeTimePredictionModel:
    def __init__(self):
        """
        モデルを初期化する
        """
        self.nqubit = 5  # qubitの数
        self.c_depth = 5  # circuitの深さ
        self.time_step = 0.79  # ランダムハミルトニアンによる時間発展の経過時間
        self.theta = None  # 学習済みパラメータを保存

    def train(self, typetimes: list[TypeTime]):
        """
        モデルを初期状態から学習する
        """
        self.words = [typetime.word for typetime in typetimes]
        self.times = [typetime.time_ms for typetime in typetimes]
        self._prepare_data()
        self._build_quantum_circuit()
        self._train_model()

    def partial_train(self, typetimes: list[TypeTime]):
        """
        モデルを追加学習する
        """
        self.words = [typetime.word for typetime in typetimes]
        self.times = [typetime.time_ms for typetime in typetimes]
        self._prepare_data()
        self._build_quantum_circuit()
        self._train_model(partial=True)

    def predict_times(self, words: list[str]) -> list[TimeMs]:
        """
        単語を打つのに必要な時間を予測する
        """
        predictions = []
        for word in words:
            encoded_word = self._my_character_encoder([word])
            normalized_word = [(2 * (x - self.min_val) / (self.max_val - self.min_val)) - 1 for x in encoded_word]
            prediction = self._predict(normalized_word[0])
            predictions.append(prediction)
        return predictions

    def _prepare_data(self):
        self.min_val = 0.0
        self.max_val = 3636363636 # データセットの単語の文字数によって変更

        self.x_train = self._my_character_encoder(self.words)
        self.y_train = np.array(self._normalize_times(self.times))
        
        self.x_train.insert(0, 0)
        self.x_train = [(2 * (x - self.min_val) / (self.max_val - self.min_val)) - 1 for x in self.x_train]
        self.y_train = np.insert(self.y_train, 0, 0)

    def _normalize_times(self, times):
        max_time = np.max(times)
        min_time = np.min(times)
        normalized_times = [2 * (time - min_time) / (max_time - min_time) - 1 for time in times]
        self.max_time = max_time
        self.min_time = min_time
        return normalized_times

    def _my_character_encoder(self, words):
        key_set = ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]
        res = []
        for word in words:
            string = ""
            for c in word:
                for i in range(3):
                    if c in key_set[i]:
                        string += str(i+1)
                        break
                string += str(key_set[i].index(c))
            res.append(int(string))
        return res

    def _build_quantum_circuit(self):
        self.state = QuantumState(self.nqubit)
        self.state.set_zero_state()
        
        self.I_mat = np.eye(2, dtype=complex)
        self.X_mat = X(0).get_matrix()
        self.Z_mat = Z(0).get_matrix()
        
        self._create_random_hamiltonian()
        self._create_time_evolution_gate()
        self._initialize_U_out()
        
        self.obs = Observable(self.nqubit)
        self.obs.add_operator(2., 'Z 0')

    def _create_random_hamiltonian(self):
        self.ham = np.zeros((2**self.nqubit, 2**self.nqubit), dtype=complex)
        for i in range(self.nqubit):
            Jx = -1. + 2. * np.random.rand()
            self.ham += Jx * self._make_fullgate([[i, self.X_mat]], self.nqubit)
            for j in range(i+1, self.nqubit):
                J_ij = -1. + 2. * np.random.rand()
                self.ham += J_ij * self._make_fullgate([[i, self.Z_mat], [j, self.Z_mat]], self.nqubit)
                
    def _create_time_evolution_gate(self):
        diag, eigen_vecs = np.linalg.eigh(self.ham)
        time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j * self.time_step * diag))), eigen_vecs.T.conj())
        self.time_evol_gate = DenseMatrix([i for i in range(self.nqubit)], time_evol_op)

    def _initialize_U_out(self):
        self.U_out = ParametricQuantumCircuit(self.nqubit)
        for _ in range(self.c_depth):
            self.U_out.add_gate(self.time_evol_gate)
            for i in range(self.nqubit):
                self.U_out.add_parametric_RX_gate(i, 2.0 * np.pi * np.random.rand())
                self.U_out.add_parametric_RZ_gate(i, 2.0 * np.pi * np.random.rand())
                self.U_out.add_parametric_RX_gate(i, 2.0 * np.pi * np.random.rand())

    def _make_fullgate(self, list_SiteAndOperator, nqubit):
        list_SingleGates = [self.I_mat] * nqubit
        for site, operator in list_SiteAndOperator:
            list_SingleGates[site] = operator
        return reduce(np.kron, list_SingleGates)

    def _train_model(self, partial=False):
        if partial and self.theta is not None:
            theta_init = self.theta
        else:
            theta_init = [self.U_out.get_parameter(i) for i in range(self.U_out.get_parameter_count())]

        result = minimize(self._cost_func, theta_init, method='Nelder-Mead')
        self.theta = result.x
        self._set_U_out(self.theta)

    def _cost_func(self, theta):
        self._set_U_out(theta)
        y_pred = [self._qcl_pred(x) for x in self.x_train]
        return ((y_pred - self.y_train) ** 2).mean()

    def _set_U_out(self, theta):
        for i, param in enumerate(theta):
            self.U_out.set_parameter(i, param)

    def _qcl_pred(self, x):
        state = QuantumState(self.nqubit)
        state.set_zero_state()
        self._U_in(x).update_quantum_state(state)
        self.U_out.update_quantum_state(state)
        return self.obs.get_expectation_value(state)

    def _U_in(self, x):
        U = QuantumCircuit(self.nqubit)
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x ** 2)
        for i in range(self.nqubit):
            U.add_RY_gate(i, angle_y)
            U.add_RZ_gate(i, angle_z)
        return U

    def _predict(self, x):
        time_normalized = self._qcl_pred(x)
        return self._denormalize_time(time_normalized)

    def _denormalize_time(self, time_normalized):
        return (time_normalized + 1) * (self.max_time - self.min_time) / 2 + self.min_time
