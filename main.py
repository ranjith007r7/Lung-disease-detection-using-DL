from flask import Flask,render_template,request,redirect,url_for
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method == 'POST':

        classes = ['Covid19', 'Lung Cancer', 'Normal', 'Pneumonia','Tuberculosis','Invalid']
        
        remedies = {'Covid19':',  Please quarantine yourself. Check for SPo2 levels periodically once it gets below 90 please admit yourself in an hospital',
        'Lung Cancer':',  Treatment depends on stage Treatments vary but may include surgery, chemotherapy, radiation therapy, targeted drug therapy and immunotherapy.',
        'Normal':',  No worries You are safe !!',
        'Pneumonia':',  Take some antibiotics prescribed by your healthcare provider',
        'Tuberculosis':', The main treatment for tuberculosis (TB) is to take antibiotics for at least 6 months. If TB has spread to your brain, spinal cord or the area around your heart, you may also need to take steroid medicine for a few weeks.Visit your doctor ASAP !!','Invalid':',  Please upload a valid CT Image'}
        

        file1 = request.files['filename']
        imgfile = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(imgfile)
        model = load_model('finalmodel.h5')
        #model = load_model('model.hdf5')
        img_ = image.load_img(imgfile, target_size=(224, 224, 3))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        prediction = model.predict(img_processed)
        index = np.argmax(prediction)
                    
        
           # or np.count_nonzero(prediction) ==4:
        result = str(classes[index]).title()
        if np.count_nonzero(prediction) == 5:
            print("Invalid Image Input")
            result = str(classes[5]).title()
 

        percentage = round(float(prediction[0][index] * 100), 2)
        rems=remedies[result]
        print(prediction)
        print(rems)
        return render_template('index.html', msg = result, src = imgfile, view = 'style=display:block', view1 = 'style=display:none',rems=rems)
    


if __name__ == '__main__':
    app.run(debug=True)
