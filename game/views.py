from django.http import JsonResponse
import uuid
from dataclasses import dataclass
from typing import Optional
import random, string
import json

from game.domain.prediction_model import TypeTimePredictionModel, TypeTime

    
@dataclass
class Session:
    datas: list[TypeTime]
    model: Optional[TypeTimePredictionModel]
    
SessionId = str
class SessionManager:
    def __init__(self) -> None:
        self.sessions = {}

    def create(self) -> SessionId:
        sid = str(uuid.uuid4())
        self.sessions[sid] = Session([], None)
        return sid
    
    def kill(self, sid: SessionId) -> bool:
        if sid in self.sessions:
            self.sessions.pop(sid)
            return True
        return False
    
    def get(self, sid: SessionId) -> Session:
        if sid not in self.sessions:
            raise Exception(f"session is not found: {sid}")
        return self.sessions[sid]

smanager = SessionManager()
total_words = [''.join(random.choice(string.ascii_lowercase) for _ in range(10)) for _ in range(100)]
    

def problems(request):
    # Front end must request this endpoint every NUM_RESPONSE_WORDS
    NUM_RESPONSE_WORDS = 2
    FIRST_NUM_RESPONSE_WORDS = NUM_RESPONSE_WORDS * 3
    """
    Send new words like below
    t=0 x====|====|====|
    t=1 -----x----|----|====|
    t=2 ----------x----|----|====|
    ...
    """
    
    try:
        if request.method == "GET":
            sid = request.GET.get('sid', 'none')
            logs = request.GET.get('logs', 'none')
            
            # First sesion
            if (sid == 'none'):
                # Newly create session and Choose random words
                sid = smanager.create()
                
                random.shuffle(total_words)
                words_for_response = total_words[:FIRST_NUM_RESPONSE_WORDS]
                return JsonResponse({"words": words_for_response, "sid": sid})
            
            session = smanager.get(sid)
            if logs == "none":
                # HACK: to kill session use this endpoint
                smanager.kill(sid)
                return JsonResponse({"sid": sid})

            logs: list[TypeTime] = json.loads(logs, parse_float=float, parse_int=int)
            assert type(logs) is list and len(logs) == NUM_RESPONSE_WORDS
            session.datas += logs
            
            if session.model is None:
                # Train a prediction model and choice top worst words
                session.model = TypeTimePredictionModel()
                model = session.model
                model.train(session.datas)
                
                preds = list(zip(total_words, model.predict_times(total_words)))
                sorted_preds = sorted(preds, key=lambda x:x[1], reverse=True)
                worst_words = [word for (word, _) in sorted_preds[:NUM_RESPONSE_WORDS]]
                print(worst_words)
                return JsonResponse({"words": worst_words, "sid": sid})
            else:
                model = session.model
                print(model)
                model.partial_train(logs)
                
                preds = list(zip(total_words, model.predict_times(total_words)))
                sorted_preds = sorted(preds, key=lambda x:x[1], reverse=True)
                worst_words = [word for (word, _) in sorted_preds[:NUM_RESPONSE_WORDS]]
                return JsonResponse({"words": worst_words, "sid": sid})
        else:
            raise Exception("invalid method")
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)
    