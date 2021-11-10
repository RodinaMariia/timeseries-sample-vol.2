"""
API for multivariate forecast.

The main function: make predictions to a several steps forward from the last date in the fitted data.
"""
import os
import logging
from config import settings
from datetime import datetime
import timelibs
from flask import Flask, jsonify, Response, request

app = Flask(__name__)
handler = logging.FileHandler(os.path.join(settings.output_directory, "multi_fc.log"))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)


@app.route("/")
def index():
    """Main page."""
    app.logger.debug("{}: Start api".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return 'API loaded.'


@app.route("/predict", methods=['GET', 'POST'])
def predict_call():
    """
    Build a forecast with a some horizon.

    :return: predictions and nomenclature's names.
    :rtype: flask.Response
    """
    # noinspection PyBroadException
    try:
        manager = timelibs.load_pickle('compound')
        h = 1 if request.args.get('h') is None else int(request.args.get('h'))
        responses = jsonify(manager.predict(h).to_json(orient='records', force_ascii=True))
        responses.status_code = 200
        print('finish prediction')
        return responses
    except Exception as e:
        app.logger.error("{}: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(e)))
        return Response(str(e), status=400)

