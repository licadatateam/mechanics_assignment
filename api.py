# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:54:52 2024

@author: carlo
"""

from flask import Flask, jsonify
import json
import main_mgo_assignment

app = Flask(__name__)

@app.route('/api/mgo/hubassign_<string:selected_date>', methods = ['GET'])
def get_mgo_assignments(selected_date):
    results = main_mgo_assignment.main(selected_date)
    # json.loads((r.content).decode('utf-8'))
    return results

if __name__ == "__main__":
    app.run()