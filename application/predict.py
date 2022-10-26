import onnxruntime as rt

from models import HousingFeatures, PredictionResult
import config as config

sess = rt.InferenceSession(config.MODEL_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def predict(data: HousingFeatures) -> PredictionResult:
    predicted = sess.run([label_name], {input_name: data.to_numpy()})[0]
    return PredictionResult(**{"Predicted": float(predicted[0][0])})