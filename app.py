import numpy as np
from skimage import io
from dash import Dash, html, Input, Output, callback
from dash.exceptions import PreventUpdate
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
import cv2
import forecast as fc

app = Dash(__name__)

#filename = 'https://raw.githubusercontent.com/plotly/datasets/master/mitochondria.jpg'

app.layout = html.Div([
    html.H6('Draw on image and press Save to show annotations geometry'),
    html.Div([
        DashCanvas(
            id='canvas',
            lineWidth=10,
            width=200,
            height=200,
        ),
    ], className="six columns"),
    #html.Div(html.Img(id='my-iimage', width=200), className="five columns"),
    html.Div(id='display-value')
    ])


@callback(Output('display-value', 'children'), Input('canvas', 'json_data'))
def update_data(string):
    if string:
        mask = parse_jsonstring(string, (200, 200))
    else:
        raise PreventUpdate
    
    escaled = cv2.resize((255 * mask).astype(np.uint8), dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    formated_array = escaled / 255
    output = fc.number_recog(formated_array)

    #return array_to_data_url(escaled)
    return output


if __name__ == '__main__':
    app.run(debug=True)