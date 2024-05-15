from django.http import JsonResponse
import tensorflow as tf
import os
import pandas as pd

def predict(request, srcni_utrip, distanca_do_konca, distanca_do_sedaj, cas_teka, tempo_povprecen_do_sedaj):
    model = tf.keras.models.load_model('models/model_v1.keras')
    tempo_povprecen_do_sedaj = float(tempo_povprecen_do_sedaj)
    X = pd.DataFrame({
        'srcni_utrip': [srcni_utrip],
        'distanca_do_konca': [distanca_do_konca],
        'distanca_do_sedaj': [distanca_do_sedaj],
        'cas_teka': [cas_teka],
        'tempo_povprecen_do_sedaj': [tempo_povprecen_do_sedaj]
    })
    X_3d = X.to_numpy().reshape(1, X.shape[1], 1)
    y = model.predict(X_3d)
    y = y[0][0]
    prediction2 = 60 / y
    minutes2 = int(prediction2)
    seconds2 = int((prediction2 - minutes2) * 60)
    return_string = f'Predvideni koncni tempo: {minutes2}:{seconds2:02d} min/km'
    data = {'message': return_string}
    return JsonResponse(data)