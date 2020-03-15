#import all required packages
import os
from flask import Flask, flash, render_template, redirect, request, url_for,send_file
import random
from werkzeug.utils import secure_filename
from numpy import pi
from fastai.vision import *
from PIL import Image

#Configure secret key and image upload folder
app = Flask(__name__)
app.config['SECRET_KEY'] = "supertopsecretprivatekey"
app.config['UPLOAD_FOLDER'] = "C:/Users/sriga/Desktop/ML/Springboard/Vegetable Classifier/VegetableClassifier/tmp"

#GET - Render the homepage to the user.
#POST - 
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']
        # if filename is empty, then assume no upload
        if image_file.filename == ' ':
            #print("image_file.filename is empty")
            flash('No file was uploaded.')
            return redirect(request.url)

        if image_file and is_allowed_file(image_file.filename):
            passed = False
            try:
                filename = generate_random_name(image_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                passed = make_thumbnail(filepath)
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)

@app.route('/about', methods=['GET'])
def about():
    if request.method == 'GET':
        # show the upload form
        return render_template('about.html')

@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])

def is_allowed_file(filename):
    """ Checks if a filename's extension is acceptable """
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext

LETTER_SET = list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

def generate_random_name(filename):
    """ Generate a random name for an uploaded file. """
    ext = filename.split('.')[-1]
    rns = [random.randint(0, len(LETTER_SET) - 1) for _ in range(3)]
    chars = ''.join([LETTER_SET[rn] for rn in rns])

    new_name = "{new_fn}.{ext}".format(new_fn=chars, ext=ext)
    new_name = secure_filename(new_name)

    return new_name


classes = ['ash gourd','asparagus','bamboo shoot','basil','beans','beetroot','bitter gourd','black raddish','bottle gourd','brinjal','broccoli','cabbage','capsicum','carrot',
           'cauliflower','celeriac','chayote','chilli','chinese artichokes','cluster beans','coconut','colocasia','coriander leaves','corn','cucumber','curry leaves','dill',
           'drumstick','dulse','elephant yam','fenugreek leaves','fiddleheads','flat beans','garlic','ginger','gooseberry','green mango','ivy gourd','kohlrabi','lemon','lime',
           'long beans','lotus root','mint','mushroom','nopal','oca','okra','onion','peas','plantain','plantain flower','plantain stem','potato','pumpkin','ramps','red chilli',
           'red raddish','ridge gourd','romanesco','shallots','snake gourd','sweet potato','tapioca','tomato','turnip','white onion','white raddish','yam', 'zuchini']

veginfo = {
            "beans" : "Green beans are French bean is one of the most popular and widely grown vegetables in India. The green immature pods are cooked and eaten as a vegetable. Immature pods are marketed fresh, frozen or canned, whole, cut or French cut. It is also  an important pulse crop, with high yielding ability as compared to gram and pea. It is grown in Maharahstra, Himachal Pradesh, Uttar Pradesh, Jammu and Kashmir and NE states.",
            "cluster beans" : "Cluster bean is an annual legume crop and one of the famous vegetable, popularly known as “Guar” in India. Cluster bean is cultivated for its green vegetables and dry pods, & as a forage crop and also cultivated for Green manure because guar planting increase subsequent crop yields, as this legume crop conserves soil nutrients. It is grown in all parts of India."
          }

learn = load_learner("C:/Users/sriga/Desktop/ML/Springboard/Vegetable Classifier/VegetableClassifier")

@app.route('/predict/<filename>')
def predict(filename):
    image_url = url_for('images', filename=filename)
    #resized_url = resize(image_url, '600x400')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = open_image(image_path)
    img.resize(299)
    pred_class,pred_idx,outputs = learn.predict(img) 
    top5classes = predict_topk_labels(outputs,5)
    veginfo = get_veg_text(top5classes)
    print("top5: ",top5classes)
   
    return render_template(
        'predict.html',
        prediction = pred_class,
        top5 = top5classes,
        info = veginfo,
        image_url=image_url    )

def predict_topk_labels(outputs,k):
    top5cat = []
    top5 = torch.topk(outputs,k)
    labels = learn.data.c2i
    key_list = list(labels.keys())
    val_list = list(labels.values())
    for i in range(len(top5[1])):
        top5cat.append(key_list[val_list.index(top5[1][i])])
    print(top5cat)
    return(top5cat)

def get_veg_text(top5classes):
    vegtext = []
    for i in range(len(top5classes)):
        try:
            if(top5classes[i]):
                vegtext.append(veginfo[top5classes[i]])
        except KeyError: 
            vegtext.append("No information available at present")
        except:
            print("Something went wrong ...")
    print(vegtext)
    return vegtext


def make_thumbnail(filepath):
    """ Converts input image to 128px by 128px thumbnail if not that size
    and save it back to the source file """
    img = Image.open(filepath)
    thumb = None
    w, h = img.size

    # if it is exactly 128x128, do nothing
    if w == 256 and h == 256:
        return True

    # if the width and height are equal, scale down
    if w == h:
        thumb = img.resize((256, 256), Image.BICUBIC)
        thumb.save(filepath)
        return True

    # when the image's width is smaller than the height
    if w < h:
        # scale so that the width is 128px
        ratio = w / 256.
        w_new, h_new = 256, int(h / ratio)
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        top, bottom = 0, 0
        margin = h_new - 256
        top, bottom = margin // 2, 256 + margin // 2
        box = (0, top, 256, bottom)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True

    # when the image's height is smaller than the width
    if h < w:
        # scale so that the height is 128px
        ratio = h / 256.
        w_new, h_new = int(w / ratio), 256
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        left, right = 0, 0
        margin = w_new - 256
        left, right = margin // 2, 256 + margin // 2
        box = (left, 0, right, 256)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True
    return False