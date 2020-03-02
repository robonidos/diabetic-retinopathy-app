from flask import Flask
import pandas as pd
import os
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
app.config['SECRET_KEY'] = '00c997d6d4c72b77e636bfe6a98a5e3c'
#app.config['UPLOADED_PHOTOS_DEST'] = os.path.join('uploads/')
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join('drapp/static/')

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app) 

from drapp import routes
