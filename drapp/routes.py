from flask import render_template
from drapp import app
from drapp.drform import DRForm
from werkzeug.utils import secure_filename
from drapp import model
import os
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

@app.route('/drform', methods=['GET', 'POST'])
def drform():
    form = DRForm()
    if form.validate_on_submit():
        photos = UploadSet('photos', IMAGES)
        filename = photos.save(form.retinafile.data)
        #file_url = photos.url(filename)   
        file_url = os.path.join('drapp/static/', filename)
        
        result_map = model.submitDetails(filename)
        form.result_predclass = result_map['predclass']
        form.result = True

    else:
        file_url = None
    return render_template('drform.html', title='DR Form', form=form, file_url=file_url)  
