from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired, FileAllowed
from drapp import pd
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class


class DRForm(FlaskForm):
    photos = UploadSet('photos', IMAGES)
    retinafile = FileField('Retina Image', validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')], description='Member’s Retina Image')
    result = False
    result_predclass = ""
    submit = SubmitField('Submit')
