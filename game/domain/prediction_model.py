from dataclasses import dataclass

TimeMs = float


@dataclass
class TypeTime:
    word: str
    time_ms: TimeMs


class TypeTimePredictionModel:
    def __init__(self):
        """
        モデルを初期化する
        """
        pass

    def train(self, typetimes: list[TypeTime]):
        """
        モデルを初期状態から学習する
        """
        pass

    def partial_train(self, typetimes: list[TypeTime]):
        """
        モデルを追加学習する
        """
        pass

    def predict_times(self, words: list[str]) -> list[TimeMs]:
        """
        単語を打つのに必要な時間を予測する
        """
        return [i * 2 for i in enumerate(words)]
