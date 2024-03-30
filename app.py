from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import re
from uuid import uuid4

app = Flask(__name__)

# Load the Artifacts
model = tf.keras.models.load_model('artifacts/model_v3_inceptionV3_200.h5')
data = pd.read_csv("artifacts/cleaned_data.csv") # here i'm reading a cleaned data frame which was cleaned while data analysis and model building
model_data_frame = pd.read_csv("artifacts/model_data.csv", index_col="name") # here i'm reading model_data file which is 0,1 matrics or categorical feature matrix or pivot table which is use to get a recommendation
with open("artifacts/model.pickle", "rb") as f:
    rec_model = pickle.load(f) # here i'm opening pickle file which have stored recommendation model
dishes_name = data.name.to_list()

# Define classes
category = {
    0: ["adhirasam", "Adhirasam"],
    1: ["aloo_gobi", "Aloo gobi"],
    2: ["aloo_matar", "Aloo matar"],
    3: ["aloo_methi", "Aloo methi"],
    4: ["aloo_shimla_mirch", "Aloo shimla mirch"],
    5: ["aloo_tikki", "Aloo tikki"],
    6: ["alu_pitika", "Alu Pitika"],
    7: ["amti", "Amti"],
    8: ["anarsa", "Anarsa"],
    9: ["ariselu", "Ariselu"],
    10: ["avial", "Avial"],
    11: ["baingan_fry", "Baingan Fry"],
    12: ["bajri_no_rotlo", "Bajri no rotlo"],
    13: ["balu_shahi", "Balu shahi"],
    14: ["bandar_laddu", "Bandar laddu"],
    15: ["basundi", "Basundi"],
    16: ["bebinca", "Bebinca"],
    17: ["bhakri", "Bhakri"],
    18: ["bhatura", "Bhatura"],
    19: ["bhindi_masala", "Bhindi masala"],
    20: ["biryani", "Biryani"],
    21: ["bisi_bele_bath", "Bisi bele bath"],
    22: ["black_rice", "Black rice"],
    23: ["bombil_fry", "Bombil fry"],
    24: ["brown_rice", "Brown Rice"],
    25: ["butter_chicken", "Butter chicken"],
    26: ["chak_hao_kheer", "Chak Hao Kheer"],
    27: ["chakali", "Chakali"],
    28: ["cham_cham", "Cham cham"],
    29: ["chana_masala", "Chana masala"],
    30: ["chapati", "Chapati"],
    31: ["chevdo", "Chevdo"],
    32: ["chhena_jalebi", "Chhena jalebi"],
    33: ["chhena_kheeri", "Chhena kheeri"],
    34: ["chhena_poda", "Chhena poda"],
    35: ["chicken_tikka", "Chicken Tikka"],
    36: ["chicken_tikka_masala", "Chicken Tikka masala"],
    37: ["chicken_varuval", "Chicken Varuval"],
    38: ["chicken_razala", "Chicken razala"],
    39: ["chikki", "Chikki"],
    40: ["chingri_bhape", "Chingri Bhape"],
    41: ["chingri_malai_curry", "Chingri malai curry"],
    42: ["chole_bhature", "Chole bhature"],
    43: ["chorafali", "Chorafali"],
    44: ["churma_ladoo", "Churma Ladoo"],
    45: ["daal_dhokli", "Daal Dhokli"],
    46: ["daal_baati_churma", "Daal baati churma"],
    47: ["daal_puri", "Daal puri"],
    48: ["dahi_vada", "Dahi vada"],
    49: ["dal_makhani", "Dal Makhani"],
    50: ["dal_tadka", "Dal tadka"],
    51: ["dalithoy", "Dalithoy"],
    52: ["dharwad_pedha", "Dharwad pedha"],
    53: ["dhokla", "Dhokla"],
    54: ["doodhpak", "Doodhpak"],
    55: ["dosa", "Dosa"],
    56: ["double_ka_meetha", "Double ka meetha"],
    57: ["dudhi_halwa", "Dudhi halwa"],
    58: ["dum_aloo", "Dum aloo"],
    59: ["farsi_puri", "Farsi Puri"],
    60: ["gajar_ka_halwa", "Gajar ka halwa"],
    61: ["gatta_curry", "Gatta curry"],
    62: ["ghevar", "Ghevar"],
    63: ["gud_papdi", "Gud papdi"],
    64: ["gulab_jamun", "Gulab jamun"],
    65: ["halvasan", "Halvasan"],
    66: ["idiappam", "Idiappam"],
    67: ["idli", "Idli"],
    68: ["imarti", "Imarti"],
    69: ["jalebi", "Jalebi"],
    70: ["jeera_aloo", "Jeera Aloo"],
    71: ["kaara_kozhambu", "Kaara kozhambu"],
    72: ["kabiraji", "Kabiraji"],
    73: ["kachori", "Kachori"],
    74: ["kadai_paneer", "Kadai paneer"],
    75: ["kadhi_pakoda", "Kadhi pakoda"],
    76: ["kajjikaya", "Kajjikaya"],
    77: ["kaju_katli", "Kaju katli"],
    78: ["kakinada_khaja", "Kakinada khaja"],
    79: ["kalakand", "Kalakand"],
    80: ["karela_bharta", "Karela bharta"],
    81: ["keerai_kootu", "Keerai kootu"],
    82: ["keerai_masiyal", "Keerai masiyal"],
    83: ["keerai_poriyal", "Keerai poriyal"],
    84: ["keerai_sadam", "Keerai sadam"],
    85: ["khakhra", "Khakhra"],
    86: ["khaman", "Khaman"],
    87: ["khandvi", "Khandvi"],
    88: ["kheer", "Kheer"],
    89: ["khichdi", "Khichdi"],
    90: ["khichu", "Khichu"],
    91: ["khorisa", "Khorisa"],
    92: ["kofta", "Kofta"],
    93: ["koldil_chicken", "Koldil Chicken"],
    94: ["kombdi_vade", "Kombdi vade"],
    95: ["kootu", "Kootu"],
    96: ["koshambri", "Koshambri"],
    97: ["koshimbir", "Koshimbir"],
    98: ["kothamali_sadam", "Kothamali sadam"],
    99: ["kuzhambu", "Kuzhambu"],
    100: ["kuzhi_paniyaram", "Kuzhi paniyaram"],
    101: ["laddu", "Laddu"],
    102: ["lassi", "Lassi"],
    103: ["lauki_ke_kofte", "Lauki ke kofte"],
    104: ["lauki_ki_subji", "Lauki ki subji"],
    105: ["lilva_kachori", "Lilva Kachori"],
    106: ["litti_chokha", "Litti chokha"],
    107: ["luchi", "Luchi"],
    108: ["lyangcha", "Lyangcha"],
    109: ["maach_jhol", "Maach Jhol"],
    110: ["mag_dhokli", "Mag Dhokli"],
    111: ["mahim_halwa", "Mahim halwa"],
    112: ["makki_di_roti_sarson_da_saag", "Makki di roti sarson da saag"],
    113: ["malapua", "Malapua"],
    114: ["masala_dosa", "Masala Dosa"],
    115: ["masor_tenga", "Masor tenga"],
    116: ["mawa_bati", "Mawa Bati"],
    117: ["methi_na_gota", "Methi na Gota"],
    118: ["mihidana", "Mihidana"],
    119: ["misi_roti", "Misi roti"],
    120: ["misti_doi", "Misti doi"],
    121: ["modak", "Modak"],
    122: ["mohanthal", "Mohanthal"],
    123: ["mushroom_do_pyaza", "Mushroom do pyaza"],
    124: ["mushroom_matar", "Mushroom matar"],
    125: ["muthiya", "Muthiya"],
    126: ["mysore_pak", "Mysore pak"],
    127: ["naan", "Naan"],
    128: ["namakpara", "Namakpara"],
    129: ["nankhatai", "Nankhatai"],
    130: ["navrattan_korma", "Navrattan korma"],
    131: ["palak_paneer", "Palak paneer"],
    132: ["paneer_butter_masala", "Paneer butter masala"],
    133: ["paneer_tikka_masala", "Paneer tikka masala"],
    134: ["pani_puri", "Pani puri"],
    135: ["panjeeri", "Panjeeri"],
    136: ["papad", "Papad"],
    137: ["paruppu_sadam", "Paruppu sadam"],
    138: ["pav_bhaji", "Pav Bhaji"],
    139: ["petha", "Petha"],
    140: ["phirni", "Phirni"],
    141: ["pindi_chana", "Pindi chana"],
    142: ["poha", "Poha"],
    143: ["pongal", "Pongal"],
    144: ["poornalu", "Poornalu"],
    145: ["pootharekulu", "Pootharekulu"],
    146: ["poriyal", "Poriyal"],
    147: ["pork_bharta", "Pork Bharta"],
    148: ["prawn_malai_curry", "Prawn malai curry"],
    149: ["puli_sadam", "Puli sadam"],
    150: ["puri_bhaji", "Puri Bhaji"],
    151: ["puttu", "Puttu"],
    152: ["qubani_ka_meetha", "Qubani ka meetha"],
    153: ["rabri", "Rabri"],
    154: ["rajma_chaval", "Rajma chaval"],
    155: ["ras_malai", "Ras malai"],
    156: ["rasabali", "Rasabali"],
    157: ["rasam", "Rasam"],
    158: ["rasgulla", "Rasgulla"],
    159: ["red_rice", "Red Rice"],
    160: ["rongi", "Rongi"],
    161: ["sabudana_khichadi", "Sabudana Khichadi"],
    162: ["sambar", "Sambar"],
    163: ["samosa", "Samosa"],
    164: ["sandesh", "Sandesh"],
    165: ["sandige", "Sandige"],
    166: ["sattu_ki_roti", "Sattu ki roti"],
    167: ["sev_khamani", "Sev khamani"],
    168: ["sev_tameta", "Sev tameta"],
    169: ["sevai", "Sevai"],
    170: ["shahi_paneer", "Shahi paneer"],
    171: ["shahi_tukra", "Shahi tukra"],
    172: ["shankarpali", "Shankarpali"],
    173: ["sheer_khurma", "Sheer khurma"],
    174: ["sheera", "Sheera"],
    175: ["shrikhand", "Shrikhand"],
    176: ["shufta", "Shufta"],
    177: ["shukto", "Shukto"],
    178: ["singori", "Singori"],
    179: ["sohan_papdi", "Sohan papdi"],
    180: ["sukhdi", "Sukhdi"],
    181: ["surnoli", "Surnoli"],
    182: ["sutar_feni", "Sutar feni"],
    183: ["tandoori_chicken", "Tandoori Chicken"],
    184: ["tandoori_fish_tikka", "Tandoori Fish Tikka"],
    185: ["thalipeeth", "Thalipeeth"],
    186: ["thayir_sadam", "Thayir sadam"],
    187: ["theeyal", "Theeyal"],
    188: ["thepla", "Thepla"],
    189: ["til_pitha", "Til Pitha"],
    190: ["turiya_patra_vatana_sabji", "Turiya Patra Vatana sabji"],
    191: ["undhiyu", "Undhiyu"],
    192: ["unni_appam", "Unni Appam"],
    193: ["upma", "Upma"],
    194: ["uttapam", "Uttapam"],
    195: ["vada", "Vada"],
    196: ["veg_kolhapuri", "Veg Kolhapuri"],
    197: ["vegetable_jalfrezi", "Vegetable jalfrezi"],
    198: ["vindaloo", "Vindaloo"],
    199: ["zunka", "Zunka"],
}

#########----------UTILITY FUNCTIIONS-------------###########
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image data if required
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array


def classify_image(image):
    input_image = preprocess_image(image)

    # Perform image classification using your custom model
    predictions = model.predict(input_image)

    # Get the top-1 result
    top1_result = np.argmax(predictions)

    # Get the class labels for the top-1 and top-3 results
    top1_label = category[top1_result]

    return top1_label

def get_recommendation(dish):

    X = model_data_frame[model_data_frame.index == dish] # here i'm getting a record of particular dish
    # here i put that record to my recommendation model and getting 13 recommended dish means n_neighbors = 13
    distance, cuisine_index = rec_model.kneighbors(X, n_neighbors = 13)  # here model return two things distance and index of dishes which is index of model_data data frame

    recommendation_result = [] # here i will append recommendation result in this list
    for c in cuisine_index.flatten(): # here i'm converting 2D array to 1D array using flatten() method of numpy
        recommended_dish = model_data_frame.index[c] # here i'm getting index name of particular index and index name is basically dish name
        if recommended_dish == dish: # here if dish name is recommended_dish then i will ignore here
            continue
        recommendation_result.append(recommended_dish) # here i'm appending a result to list
    
    return recommendation_result[:12]


def get_recipe(dish):
    print(dish)
    return data.loc[data.name == dish, ["recipe"]].values[0][0]


def start_case(input_string):
    # Replace underscores and hyphens with spaces
    modified_string = re.sub(r'[_-]', ' ', input_string)
    return modified_string

#ALL ROUTES

#######---------HOMEPAGE--------########
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        code_name = classify_image(file.stream)[0]

        return jsonify({'name': code_name})

    return render_template('index.html')


#######---------RECIPE--------########
@app.route('/recipe/<name>', methods=['GET'])
def recipe(name):
    if start_case(name) not in dishes_name: # here if dish name didn't get then by default it will be all and it will redirect to home page
        return redirect("/404")
    
    recommended_dishes = get_recommendation(name) # here i'm getting recommendation for that dish
    mod_recommended_dishes = list(map(start_case, recommended_dishes))
    recipe_id = get_recipe(start_case(name)) # here i'm getting unique id of youtube recipe for particular dish

    return render_template("recipe.html", current_dish = start_case(name), recommended_dishes=mod_recommended_dishes, recipe_id = recipe_id)
  


#######---------404--------########
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404



if __name__ == '__main__':
    app.run(debug=False,)
