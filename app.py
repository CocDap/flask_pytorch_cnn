import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import request, Flask, redirect, url_for, render_template
import config
import time
from PIL import Image
import numpy as np
import re
import torch
from imageio import imread
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import model
import os
import base64

random_seed = 1




app = Flask(__name__)



def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    #print(imgstr)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
    #get the raw data format of the image
    imgData = request.get_data()
    #encode it into a suitable format
    convertImage(imgData)
    #read the image into memory
    #x = imread('output.png')
    x = Image.open('output.png','r')
    x = x.convert('L')
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    #make it the right size
    x = resize(x,(28,28))
    #np.resize(x,(28,28))
    #convert to a 4D tensor to feed into our model
    x = np.expand_dims(x,axis =0)
    x = x[np.newaxis,:,:]
    #np.save('test4',x)
    x = torch.from_numpy(x).float()
    with torch.no_grad():
        out = _model(x)
        _, predicted = torch.max(out.data, 1)
        predicted = predicted.data[0].item()
        return str(predicted)





if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _model = model.Net()
    _model.load_state_dict(torch.load('./results/model.pth'))
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug =True)

