import streamlit as st
import pickle
import pandas as pd
from streamlit_pills import pills


# ---------------Page Setup-----------------#

st.set_page_config(layout="centered", page_title="Used Cars Estimator")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Set Title
st.title(":blue[Used Cars Price Estimator]")
st.subheader(":red[Using Machine Learning]")
st.markdown("#")

# --------------- Slider Section-----------------------------
st.sidebar.markdown("## :blue[Ⓜ️Select Model]")

with st.sidebar:
    selected_model = pills("Models", ["Decision Tree", "Gradient Boosting"], ["🌳", "💥"])
st.markdown("#")
load_model_button = st.sidebar.button("Load Model")
# st.sidebar.markdown("---")
st.sidebar.markdown(f" ## :blue[ℹ️ Dataset Infomation]")
# st.sidebar.write("Description of the dataset's features are given below")
 

with st.sidebar.expander("Read more about dataset features"):
    st.markdown(
        "1️⃣ **Horsepower:** Horsepower is the power produced by an engine  \n"
        "2️⃣ **Mieage:** The number of miles that it can travel using one gallon or litre of fuel  \n"
        "3️⃣ **Fuel_tank_volume:** Fuel tank's filling capacity in gallons  \n"
        "4️⃣ **Model_name:** Model name of the vehicle  \n"
        "5️⃣ **height:** Height of the vehicle in inches  \n"
        "6️⃣ **franchise_make:** The company that owns the franchise.  \n"
        "7️⃣ **Make Name:** Vehicle's make name  \n"
        "8️⃣ **Wheelbase:** Wheelbase of the vehicle  \n"
        "9️⃣ **Daysonmarket:** Days since the vehicle was first listed on the website  \n"
        "🔟 **Width:** The vehicle's width  \n"
        "🔟 **Power:** The vehicle's power  \n"
        "🔟 **Engine Displacement:** The measure of the cylinder volume swept by all of the pistons of a - - piston engine, excluding the combustion chambers.  \n"
    )
st.markdown("---")


# Fuction for loading and caching the model
@st.cache_resource
def load_data():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    GBR_model = pickle.load(open("GBR_model.pkl", "rb"))
    DT_model = pickle.load(open("DT_model.pkl", "rb"))
    
    return GBR_model, DT_model, scaler


# Load model
if load_model_button:
    GBR_model, DT_model, scaler = load_data()
    if "GBR_model" not in st.session_state:
        st.session_state["GBR_model"] = GBR_model
    if "DT_model" not in st.session_state:
        st.session_state["DT_model"] = DT_model
    if "scaler" not in st.session_state:
        st.session_state["scaler"] = scaler


##### FEATURES ######

wheel_system_map = {'FWD': 3, 'AWD': 2, 'RWD': 4, '4WD': 0, '4X2': 1}

franchise_make_map = {'Jeep': 20, 'Land Rover': 23, 'FIAT': 11, 'Chevrolet': 8, 'Cadillac': 7, 'Chrysler': 9, 'Dodge': 10, 'RAM': 36, 'Ford': 13, \
                'Kia': 21, 'Mazda': 29, 'Audi': 3, 'Hyundai': 17, 'Toyota': 41, 'Lincoln': 25, 'Volvo': 43, 'GMC': 14, 'Volkswagen': 42, 'BMW': 4, 
                'Lexus': 24, 'Buick': 6, 'Subaru': 40, 'Scion': 39, 'Honda': 16, 'Nissan': 33, 'Acura': 0, 'INFINITI': 18, 'Porsche': 35, 'Rolls-Royce': 37, 
                'Bentley': 5, 'Lamborghini': 22, 'Mercedes-Benz': 31, 'Jaguar': 19, 'Maserati': 28, 'Alfa Romeo': 1, 'Ferrari': 12, 'MINI': 27, 'Mitsubishi': 32, 
                'Aston Martin': 2, 'Lotus': 26, 'McLaren': 30, 'Pagani': 34, 'Genesis': 15, 'SRT': 38, 'smart': 44}

make_name_map = {'Jeep': 29, 'Land Rover': 34, 'Subaru': 62, 'Mazda': 42, 'Alfa Romeo': 3, 'BMW': 7, 'Hyundai': 25, 'Chevrolet': 11, 'Lexus': 35, 
             'Cadillac': 10, 'Chrysler': 12, 'Dodge': 14, 'Mercedes-Benz': 44, 'Nissan': 47, 'Honda': 22, 'Kia': 32, 'Ford': 18, 'Lincoln': 36,
             'Audi': 5, 'Jaguar': 28, 'Volkswagen': 68, 'RAM': 54, 'Porsche': 53, 'Toyota': 66, 'INFINITI': 26, 'GMC': 20, 'Acura': 2, 'Maserati': 40, 
             'FIAT': 15, 'Volvo': 69, 'Mitsubishi': 46, 'Buick': 9, 'Mercury': 45, 'Scion': 58, 'Saab': 56, 'MINI': 39, 'Ferrari': 16, 'Genesis': 21, 'Saturn': 57, 
             'Bentley': 8, 'Suzuki': 64, 'Tesla': 65, 'Fisker': 17, 'Pontiac': 52, 'Lamborghini': 33, 'smart': 71, 'Hummer': 24, 'Rolls-Royce': 55, 'Lotus': 37, 'Spyker': 60, 
             'McLaren': 43, 'Aston Martin': 4, 'Kaiser': 30, 'Oldsmobile': 48, 'Maybach': 41, 'Freightliner': 19, 'Karma': 31, 'Isuzu': 27, 'Plymouth': 51, 'Shelby': 59, 'Triumph': 67, 
             'MG': 38, 'Pagani': 49, 'Datsun': 13, 'Studebaker': 61, 'AM General': 0, 'Austin-Healey': 6, 'AMC': 1, 'Hudson': 23, 'Willys': 70, 'Pininfarina': 50, 'Sunbeam': 63}

model_name_map = {'Renegade': 829, 'Discovery Sport': 319, 'WRX STI': 1062, 'Discovery': 317, 'Range Rover Velar': 818, 'MAZDA3': 609, 'Range Rover Evoque': 814, 
              '4C': 44, '3 Series': 16, 'CX-3': 213, 'CX-5': 215, 'Elantra': 346, 'Malibu': 631, 'RC 350': 785, 'Traverse': 1015, 'Grand Cherokee': 501, 'Compass': 261, 
              'Impreza': 534, 'Veloster': 1042, 'XT4': 1092, '200': 7, 'Equinox': 367, 'Range Rover Sport': 817, 'Wrangler Unlimited': 1066, 'Durango': 320, 'CLA-Class': 195, 
              'Charger': 244, 'Silverado 1500': 922, 'Rogue': 841, 'Civic': 249, 'RX 350': 802, 'GLC-Class': 459, 'Optima': 698, 'Explorer': 383, 'Navigator': 685, 'Outback': 701, 
              '4 Series': 35, 'GLE-Class': 460, 'Escalade': 369, 'Tucson': 1020, '2 Series': 6, 'Pacifica': 709, 'Suburban': 963, 'Camaro': 223, 'Cruze': 293, 'Trax': 1016, 'SQ5': 875, 
              'E-PACE': 323, 'Maxima': 645, 'Cherokee': 245, 'C-Class': 174, 'Tiguan': 996, 'Range Rover Hybrid': 815, 'MAZDA6': 611, 'Defender': 312, 'F-350 Super Duty': 398, 'Colorado': 259, 
              'Pathfinder': 719, 'Range Rover': 813, 'Tahoe': 984, 'CX-9': 217, 'Blazer': 163, 'Sorento': 943, 'SRX': 878, '1500': 3, 'Mustang': 672, 'X5': 1074, 'Challenger': 243, 
              'Impala': 532, 'Cayenne': 237, 'F-PACE': 405, 'GLK-Class': 462, '5 Series': 46, 'Silverado 2500HD': 925, 'Pilot': 726, 'E-Series': 324, 'R8': 768, 'Town & Country': 1004, 
              'X6 M': 1077, 'Altima': 123, 'Transit Connect': 1012, 'X3': 1070, 'GTI': 477, '7 Series': 70, 'E-Class': 322, 'A4': 96, 'Rogue Sport': 844, 'Soul EV': 945, 
              'IS 350': 531, 'WRX': 1061, 'Prius': 730, 'Sonata': 938, 'Panamera Hybrid': 715, 'Grand Caravan': 500, 'Patriot': 721, 'Santa Fe': 887, 'GL-Class': 456, 
              'LS 460': 570, 'X4': 1072, 'X1': 1068, 'Accord': 114, 'Civic Coupe': 250, 'CLS-Class': 197, 'G35': 448, 'Passat': 717, 'Sprinter': 954, 'Q5': 744, 'Canyon': 228, 
              'Journey': 549, 'MDX': 614, 'A5': 99, 'Volt': 1059, 'Tacoma': 983, 'XT5': 1093, 'Titan': 997, 'A3': 94, 'M-Class': 596, 'Forte': 434, 'Levante': 591, 'Escalade ESV': 370, 
              'G-Class': 445, '6 Series': 62, 'F-150': 392, 'ATS': 108, 'GLA-Class': 457, 'Savana Cargo': 893, 'X6': 1076, 'RSX': 798, 'Liberty': 592, 'i8': 1117, 'Panamera': 713, 
              'RAM 1500': 770, 'Camry': 224, 'Yukon XL': 1103, 'S-Class': 848, 'Terrain': 990, 'S-Class Coupe': 849, 'CTS': 207, 'Edge': 341, 'XV Crosstrek': 1096, 'Acadia': 112, 'Q70': 751, 
              'Fiesta': 422, 'Sportage': 953, 'RAV4': 779, 'Escape': 373, 'Silverado 3500HD': 927, 'Sedona': 895, 'Odyssey': 696, 'S6': 860, 'SL-Class': 870, 'MKZ Hybrid': 623, 'SLK-Class': 872, 
              'Golf': 492, '124 Spider': 2, 'Soul': 944, 'GS 350': 467, 'CTS-V': 210, 'Santa Fe Sport': 888, 'Niro': 690, 'Q60': 749, 'Sierra 3500HD': 914, 'Jetta': 544, 'Fusion': 442, 
              'Prius Prime': 732, 'Yukon': 1101, 'TSX': 979, 'Forester': 433, 'Focus': 429, 'Arteon': 129, 'S80': 866, 'Highlander Hybrid': 517, 'MKZ': 622, 'Sierra 1500': 906, 'XC70': 1082,
            'V60': 1032, 'Prius v': 734, 'Rio5': 835, 'Mirage': 652, 'Nitro': 693, 'Yaris': 1099, '2500': 13, 'Lucerne': 593, 'Murano': 669, 'Crosstrek': 290, 'Taurus': 986, 'Corolla': 273, 
            'Xterra': 1098, '4Runner': 45, 'Cadenza': 219, 'Elantra GT': 348, 'Eclipse Cross': 336, 'HR-V': 514, 'Grand Marquis': 502, 'Sienna': 905, 'Legacy': 589, 'XTS': 1095, 'RDX': 787, 
            'Commander': 260, 'Wrangler': 1065, 'ProMaster': 736, 'CR-V': 198, 'Regal': 823, 'MKC': 618, 'Rogue Select': 843, 'ProMaster City': 738, 'Ridgeline': 833, 'C70': 188, 'TLX': 977, 'EX35': 334,
            'XT6': 1094, 'Highlander': 516, 'QX50': 756, 'Impala Limited': 533, 'Fit': 424, 'DTS': 307, 'K5': 551, 'Seltos': 896, 'Enclave': 355, 'Armada': 127, 'IS 250': 530, 'New Yorker': 687, 
            'Milan': 649, 'Sequoia': 899, 'Transit Passenger': 1014, 'Magnum': 630, 'FJ Cruiser': 413, 'Routan': 846, 'Expedition': 382, 'ES 350': 332, 'G37': 449, 'El Camino': 344, 'XC90': 1083, 
            'Avenger': 144, 'R-Class': 764, 'Golf SportWagen': 495, 'xA': 1122, 'Niro Hybrid Plug-In': 692, 'S-TYPE': 851, 'Trailblazer': 1007, 'Jetta SportWagen': 547, 'Accord Coupe': 115, 
            'Cobalt': 257, 'Stinger': 960, 'Niro EV': 691, 'Matrix': 644, 'Rio': 834, 'Telluride': 989, 'Gladiator': 491, 'Q7': 750, '300': 18, 'Accord Hybrid': 117, 'Fusion Hybrid': 444, 
            'Golf Alltrack': 493, 'CT4': 202, 'CT5': 203, 'PT Cruiser': 708, 'TL': 976, '9-5': 82, 'Avalon': 142, 'Palisade': 712, 'X-TYPE': 1067, 'XJ-Series': 1088, 'ES 300': 329, 
            'M3': 598, 'S40': 857, 'Ranger': 819, 'Atlas': 136, 'Eclipse': 335, 'M4': 603, 'GLS-Class': 463, '350Z': 29, 'Metris': 646, 'Ghibli': 488, 'Cooper': 268, 'S4': 855, 'Malibu Maxx': 633, 
            '3 Series Gran Turismo': 17, 'Bolt EV': 164, 'A8': 104, 'Express Cargo': 388, '3500': 27, 'CT6': 204, '5 Series Gran Turismo': 47, 'A6': 101, 'QX30': 754, 'QX60': 758, 
            'Mustang Shelby GT350': 675, 'QX80': 761, 'M5': 604, 'RS 7': 796, 'Sentra': 897, 'S3': 854, 'Corvette': 282, 'Genesis Coupe': 487, 'Q3': 741, 'F-250 Super Duty': 396, 
            '6 Series Gran Turismo': 63, '5500 Chassis': 55, 'MDX Hybrid Sport': 615, 'Macan': 629, '1 Series': 0, 'Kona': 557, 'Encore': 356, 'Cruze Limited': 294, 'Elantra Touring': 
            349, 'X5 M': 1075, 'CX-30': 214, 'LX 570': 580, 'S7': 863, 'Cooper Clubman': 269, 'MKX': 621, 'ILX': 525, 'MKS': 619, 'S60': 861, 'MAZDA5': 610, 'S5': 858, 'MX-5 Miata': 628, 
            'Golf R': 494, 'Altima Hybrid': 125, 'Lancer': 583, 'Ioniq Electric': 539, 'CC': 190, 'C/K 1500': 179, 'Outlander': 702, 'MKT': 620, 'F430': 410, 'Beetle': 157, 
            'CT Hybrid': 201, 'Element': 352, 'Continental': 263, 'A7': 103, 'G80': 454, 'Zephyr': 1110, 'A5 Sportback': 100, 'SC 430': 869, 'QX70': 760, 'Veloster Turbo': 1044, 
            'Versa': 1051, 'Eldorado': 350, 'Tundra': 1021, 'RAM 2500': 772, 'Focus RS': 431, 'G90': 455, 'Azera': 148, 'A4 Avant': 98, 'Caprice': 230, 'Accent': 113, 'GranTurismo': 498, 
            'Galant': 482, 'Civic Hatchback': 251, 'ION': 527, 'RAV4 Hybrid': 780, 'STS': 881, 'M6': 606, 'TTS': 982, 'Sprinter Cargo': 955, 'RX-8': 806, 'C-Max Hybrid': 177, 'AMG GT': 106, 
            'Sonic': 941, 'Sebring': 894, 'EcoSport': 338, 'RX 330': 801, 'A-Class': 93, 'TT': 980, 'Venue': 1047, 'Corolla Hybrid': 275, 'Cube': 295, 'Versa Note': 1052, 'Quattroporte': 762, 
            'CTS Coupe': 208, 'Sonata Hybrid': 939, '9-3': 79, 'S5 Sportback': 859, '500': 48, 'NX 200t': 682, 'A4 Allroad': 97, 'Transit Cargo': 1010, 'Bentayga': 160, 'CR-Z': 200, 'SX4': 883, 
            'Stratus': 961, 'Model S': 656, 'Karma': 553, 'Camry Hybrid': 225, 'Forte5': 436, 'X3 M': 1071, 'Ioniq Hybrid': 540, 'M37': 602, 'X7': 1078, 'HHR': 513, 'XC60': 1081, 'Corsair': 278, 
            'V90': 1035, 'XC40': 1080, 'NV200': 680, 'M2': 597, 'X4 M': 1073, 'C-HR': 175, 'G8': 453, 'QX60 Hybrid': 759, 'Cayenne E-Hybrid': 238, 'Nautilus': 684, 'Prius c': 733, 'Q50': 747, 
            'Mustang Shelby GT500': 676, 'Passport': 718, 'RX': 799, 'Q70L': 752, 'G70': 452, 'Frontier': 440, 'GX': 480, '718 Cayman': 72, 'Q8': 753, 
            'Flex': 427, 'Mark VIII': 642, 'Avalanche': 141, 'Dakota': 308, 'Sierra 2500HD': 911, 'S90': 867, 'Aviator': 147, 'Mirage G4': 653, 'Countryman': 284, 
            'Verano': 1050, 'Outlander Sport': 704, 'Avalon Hybrid': 143, '86': 76, 'California T': 222, 'Voyager': 1060, 'Optima Hybrid': 699, 'Express': 387, 'Veloster N': 1043, 
            'Outlander Hybrid Plug-in ': 703, 'Kona Electric': 558, 'Outlook': 705, 'Sonata Hybrid Plug-In ': 940, 'Vibe': 1053, 'Grand Prix': 503, 
            'Jetta Hybrid': 546, 'RL': 788, 'Huracan': 521, 'BRZ': 154, 'Ioniq Hybrid Plug-In ': 541, 'Town Car': 1005, 'ELR': 327, 'Genesis': 486, 'RX 300': 800, 'Five Hundred': 425, 
            'Envoy': 361, 'Crosstour': 289, 'Jetta GLI': 545, 'Impreza WRX': 535, 'Touareg': 1001, 'Corolla Hatchback': 274, 'Forte Koup': 435, 'Q40': 742, 'LR2': 565, 'Giulia': 490, 
            'F-450 Super Duty': 401, 'Fusion Energi': 443, 'FX35': 415, 'Model X': 658, 'RX 400h': 803, 'Stelvio': 959, 'IS': 529, 'C-Max Energi': 176, 'Atlas Cross Sport': 137,
            '911': 86, 'fortwo': 1113, 'XF': 1085, 'Continental GTC': 266, 'Model 3': 654, 'Corolla iM': 276, 'Camry Solara': 226, 'XE': 1084, 'LEAF': 563, 'X2': 1069, 
            'Escape Hybrid': 374, 'QX56': 757, 'Venza': 1048, 'LaCrosse': 581, 'Yaris iA': 1100, 'Tahoe Hybrid': 985, 'LR4': 567, 'Juke': 550, 'ES': 328, 'tC': 1121, '8 Series': 74, 
            'VUE': 1036, 'M8': 607, 'Explorer Hybrid': 384, 'F-350 Super Duty Chassis': 399, 'Impreza WRX STI': 536, 'Ascent': 131, 'Transit Crew': 1013, 'Silverado Classic 2500HD': 930, 
            'LX 470': 579, 'Silverado Classic 1500': 929, 'F-550 Super Duty Chassis': 404, 'Lancer Evolution': 584, 'NX': 681, 'Savana': 892, 'Crosstrek Hybrid': 291, 'RX Hybrid': 804,
            'Silverado Classic 3500': 931, 'Optima Hybrid Plug-In ': 700, 'Supra': 972, 'Focus Electric': 430, 'RAM 3500': 774, 'i3': 1116, 'Envision': 360, 'Spark': 946, 
            'Continental GT': 265, 'Excursion': 380, '944': 91, 'Silverado 3500': 926, 'NV Cargo': 678, 'Escalade EXT': 371, 'B-Class': 151, '458 Italia': 42, 'RS 3': 791, 
            'e-Golf': 1111, 'Clarity Hybrid Plug-In ': 255, 'Z4': 1106, 'Boxster': 167, 'SLC-Class': 871, 'GLB-Class': 458, 'Altima Coupe': 124, 'Santa Fe XL': 889, 'Cascada': 234, 
            'LS 500': 571, 'F-TYPE': 406, 'LC': 561, 'Mulsanne': 668, 'xD': 1124, 'Endeavor': 358, 'Quest': 763, 'iA': 1118, 'H3': 511, 'Amanti': 126, 'Envoy XL': 362, 'Caliber': 220,
                'CX-7': 216, 'H2': 509, 'Aveo': 146, 'Pacifica Hybrid': 710, 'Dart': 309, 'Veracruz': 1049, '500X': 51, 'Spectra': 950, 'Sable': 884, 'Encore GX': 357, 'JX35': 542, 
                'Rondo': 845, 'Sonoma': 942, 'Elantra Coupe': 347, 'ATS Coupe': 109, 'M35': 600, 'GS 300': 466, 'Ghost': 489, 'Captiva Sport': 231, 'Explorer Sport Trac': 386, 
                'Q50 Hybrid': 748, 'Kicks': 555, 'MAZDA2': 608, 'xB': 1123, 'Elise': 353, 'iQ': 1120, 'Aura': 138, 'Mariner': 636, 'Malibu Hybrid': 632, 'Land Cruiser': 586, 'Cabrio': 218, 
                'XC': 1079, 'Grand Vitara': 505, 'Tribute': 1018, 'G6': 451, 'V50': 1031, '3500 Chassis': 28, 'ECHO': 326, 'Protege': 739, 'Aerio': 119, 'Econoline Cargo': 339, 
                'Accord Crosstour': 116, 'Transit Chassis': 1011, 'F-450 Super Duty Chassis': 402, 'Freestar': 438, 'GX 470': 481, 'ES 300h': 330, 'Metris Cargo': 647, 'Murano CrossCabriolet': 
                670, 'Sephia': 898, 'NX Hybrid': 683, 'ES Hybrid': 333, 'LX': 577, 'RC': 782, 'HS 250h': 515, 'Mustang SVT Cobra': 674, 'UX Hybrid': 1025, 'FX37': 416, 'ES 330': 331, 
                'Insight': 537, 'S8': 865, 'C8': 189, 'Dawn': 310, 'Phantom Drophead Coupe': 724, 'SLR McLaren': 873, '250 GT': 12, 'Phantom': 723, 'Urus': 1027, 'Flying Spur': 428,
                    'Wraith': 1064, 'A3 Sportback': 95, '718 Boxster': 71, '912': 87, 'Panamera E-Hybrid': 714, '570S': 59, 'RLX': 789, 'RLX Hybrid Sport': 790, '9-3 SportCombi': 80, 
                    'Astro Cargo': 135, 'Rapide': 821, 'B9 Tribeca': 153, 'CR-V Hybrid': 199, 'Monte Carlo': 662, 'Tribeca': 1017, 'MP4-12C': 624, 'Astro': 134, 'Safari': 885, 
                    'Safari Cargo': 886, 'LR3': 566, 'Mariner Hybrid': 637, 'C/V': 186, 'Century': 242, '370Z': 33, 'CL-Class': 194, 'NV Passenger': 679, 'FR-S': 414, 
                    'S-Series': 850, 'Civic Type R': 253, 'Sunfire': 964, 'Celica': 241, 'GT-R': 474, 'E-Series Chassis': 325, 'Crossfire': 287, 'Bel Air': 158, '190-Class': 5, 
                    'Z3': 1104, 'Continental Flying Spur': 264, 'Corvair': 281, 'RC F': 786, 'Manhattan': 634, 'S-10': 847, 'LS': 568, 'Special': 948, 'Toronado': 999, 
                    'RAM 150': 769, 'Chevelle': 246, 'California': 221, 'Fairlane': 419, 'Thunderbird': 993, 'Baja': 155, 'Sky': 935, 'LeSabre': 588, 'Aventador': 145, 'F12 Berlinetta': 407, 
                    'Roadmaster': 838, 'Z4 M': 1107, 'Rabbit': 807, 'XLR': 1091, 'Trailblazer EXT': 1008, 'Portofino': 728, '812 Superfast': 75, 'GTO': 478, '488': 43, 'Viper': 1056, 'R32': 767, 
                    'Mark VII': 641, 'RS Q8': 797, 'GTC4Lusso': 475, '599 GTB Fiorano': 61, 'GTC4Lusso T': 476, 'Civic Hybrid': 252, 'XK-Series': 1089, 'Eos': 365, 'Taycan': 988, 'Escalade Hybrid': 372, 
                    'RAV4 Prime': 781, 'S2000': 853, 'RS 5': 793, 'Tiburon': 994, 'RS 6': 795, 'Touareg 2': 1002, '500L': 50, 'Mark LT': 639, 'Cooper Paceman': 271, 'XL-7': 1090, 
                    'CLK-Class': 196, 'DeVille': 311, 'e-tron': 1112, 'A8 Hybrid Plug-In': 105, 'A6 Allroad': 102, 'Q5 Hybrid Plug-in': 746, 'TT RS': 981, 'Sierra 3500': 913, 
                    'Delta 88': 313, 'Nova': 695, 'Firebird': 423, 'Cayman': 240, 'Vigor': 1054, 'Legend': 590, 'Riviera': 836, 'DB7': 304, 'Sierra 2500HD Classic': 912, 
                    'Crown Victoria': 292, 'Entourage': 359, 'XV Crosstrek Hybrid': 1097, 'GranSport': 497, 'G25': 447, 'LS 600h L': 573, 'Silverado 3500HD Chassis': 928, 'SS': 879, 
                    '57': 57, 'RC 300': 784, '4500 Chassis': 41, 'F-550 Super Duty': 403, 'Grand Am': 499, 'G20': 446, 'Equator': 366, 'ILX Hybrid': 526, 'Uplander': 1026,
                    'Model A': 655, 'DB9': 305, 'Roadster': 839, 'Astra': 133, 'SSR': 880, 'Pathfinder Hybrid': 720, 'V8 Vantage': 1034, 'C30': 187, 'ATS-V': 110, 'Brougham': 173,
                    'LTD Crown Victoria': 576, 'Bronco': 169, 'I-PACE': 522, 'GS F': 471, 'Super Beetle': 967, 'Reatta': 822, 'MAZDASPEED3': 612, 'Taurus X': 987, 'Range Rover Hybrid Plug-in': 816, 
                    'RX-7': 805, 'Cullinan': 296, 'K900': 552, 'FF': 412, 'City Express': 248, 'Vanquish': 1040, 'DBS': 306, 'Evora': 379, 'Revero GT': 832, 'Vantage': 1041, 'DB11': 303, 'Econoline Wagon': 340,
                    'Fleetwood': 426, 'Prius Plug-In': 731, 'QX4': 755, 'Cavalier': 236, '500e': 52, 'Windstar': 1063, 'SQ7': 876, 'Freelander': 437, 'Mountaineer': 667, 'H3T': 512, '356': 30, 'Silver Spur': 921, 
                    'RAM 3500 Chassis ': 775, 'UX': 1024, 'GS': 464, 'Rogue Hybrid': 842, 'C/K 10': 178, 'I35': 524, 'Montego': 663, 'V70': 1033, 'Equus': 368, '550': 54, 'ATS-V Coupe': 111, 'Mustang Mach-E': 673, 
                    '430 Scuderia': 38, 'CTS-V Coupe': 211, 'Trooper': 1019, '328': 25, 'Freestyle': 439, 'Capri': 229, 'Express Chassis': 389, 'MPV': 625, 'SQ8': 877, 'Carrera GT': 233, 'Eclipse Spyder': 337,
                    'Continental Supersports': 267, '250': 11, 'SLS-Class': 874, 'Rendezvous': 828, '308': 23, 'LC Hybrid': 562, 'Alero': 120, 'Series III': 902, 'Gallardo': 485, 'Elan': 345, 'H2 SUT': 510, '720S': 73,
                    'Cayenne Hybrid': 239, 'CTS Sport Wagon': 209, 'iM': 1119, 'Escort': 376, '9-7X': 84, 'Yukon Hybrid': 1102, 'Concorde': 262, 'Caravan': 232, 'LS 500h': 572, 'RAM Van': 777, 'Allante': 121, 'Torrent': 1000, 
                    'V12 Vanquish': 1028, 'Barracuda': 156, 'Aztek': 149, 'Monterey': 664, 'Sierra 3500HD Chassis': 915, 'Borrego': 166, 'RS 4': 792, 'C/K 3500': 184, 'Park Avenue': 716, 'Fiero': 421, '928': 90, 'Prowler': 740, 
                    'Electra': 351, 'Sierra 1500 Limited': 908, 'LS 430': 569, 'L-Series': 559, 'Venture': 1046, '380-Class': 34, 'Cooper Coupe': 270, 'MR2 Spyder': 627, 'Sierra Classic 1500': 916, 'Sierra 2500': 910, 'Countryman Hybrid Plug-in ': 285, 
                    'G5': 450, '570GT': 58, 'I30': 523, 'Seville': 903, 'Rainier': 809, 'Coupe': 286, '360 Spider': 32, '500-Class': 49, 'Brooklands': 171, 'LX 450': 578, 'Lagonda': 582, 'VUE Hybrid': 1037, 'Custom Cruiser': 298, 'Turbo R': 1022, 'Cobra': 258, 
                    'C/K 2500 Series': 183, 'C/K 2500': 182, '240': 10, 'Corsica': 279, 'RAM 350': 773, 'Bonneville': 165, 'F-450': 400, 'Forenza': 432, 'C/K 20': 181, 'Grand Ville': 504, 'L300': 560, '900': 85, '360': 31, 
                    'F430 Spider': 411, '600LT': 65, '650S': 68, 'RS 5 Sportback': 794, 'GT': 473, '675LT': 69, 'Superamerica': 969, '924': 89, 'TR6': 978, 'Mondial': 659, '914': 88, '575M': 60, '612 Scaglietti': 66, 'Aspen': 132, 'Kizashi': 556, 'CJ-5': 191,
                    'Touareg Hybrid': 1003, 'Ramcharger': 811, 'S4 Avant': 856, 'XF Sportbrake': 1086, 'Solstice': 937, 'Neon': 686, '960': 92, 'Grand Wagoneer': 507, '442': 39, 'CTS-V Wagon': 212, 'CT6-V': 206, 'Lumina': 594, 'Testarossa': 992, 'Stealth': 958, '280': 14,
                    'Silverado 1500HD': 923, 'Mark VI': 640, '3100': 24, 'Phaeton': 722, 'Allroad': 122, '9-4X': 81, 'ZDX': 1109, 'Spark EV': 947, '420-Class': 37, '560-Class': 56, '300ZX': 22, 'Sierra 1500 Hybrid': 907, 'C/K 3500 Series': 185, 'Silhouette': 918,
                    'Bravada': 168, 'CL': 193, 'Silverado Hybrid': 932, 'NSX': 677, 'Terraza': 991, 'Regal Sportback': 824, 'Deluxe': 314, 'Valiant': 1038, 'Super Deluxe': 968, 'Tracker': 1006, 
                    'MGB': 617, '300-Class': 19, 'RC 200t': 783, 'GS 460': 470, 'Regal TourX': 825, 'Le Baron': 587, 'Classic': 256, 'Aurora': 140, 'fortwo electric drive': 1114, 'RAM 4500 Chassis ': 776, 'Ranchero': 812, 'Satellite': 891, 'Bentayga Hybrid': 161, 'LFA': 564, 
                    'F12tdf': 408, 'Huayra': 519, 'Enzo': 364, '8C Competizione': 77, 'ActiveHybrid 7': 118, 'Prelude': 729, 'Prizm': 735, 'Relay': 827, 'P1': 706, 'Montero Sport': 666, 'Diplomat': 316, 'XG350': 1087,
                    'F-150 Heritage': 393, '164': 4, 'Galaxie 500': 484, 'EuroVan': 378, 'R/V 3500': 766, 'F-350': 397, '450-Class': 40, 'Cortina': 280, 'ProMaster Chassis': 737, 
                    'Exige': 381, 'Murcielago': 671, 'Superbird': 970, '512TR': 53, 'Diablo': 315, 'Regency': 826, 'Montana SV6': 661, 'Cutlass Supreme': 302, 'Marauder': 635, '400': 36, 'GS Hybrid': 472, 'Jimmy': 548, '300M': 21, 'Arnage': 128, 'H1': 508, 'SC 400': 868, 'Vitara': 1058, 'LS Hybrid': 574, 'B-Series': 152, 'S70': 864, 'M56': 605, 'Torino': 998, 'V12 Vantage': 1029, 'Revero': 831, 'Mighty Max Pickup': 648, '280ZX': 15, 'Rodeo': 840, 'Explorer Sport': 385, 'MGA': 616, 'Cougar': 283, 'F-100': 391, 'Bronco Sport': 170, 'Envoy XUV': 363, 'Silverado 2500': 924, 'F-150 SVT Lightning': 394, 'M30': 599, '3000GT': 20, 'Custom': 297, 'Lancer Sportback': 585, 'Sierra 1500HD': 909, 'Montana': 660, 'Blackwood': 162, 'GLI': 461, '9-2X': 78, 'Newport': 688, 'F355': 409, 'F-250': 395, 'Galaxie': 483, 'Raider': 808, 'Model T': 657, 'Karmann Ghia': 554, 'C/K 1500 Series': 180, 'Pickup': 725, 'Escape Hybrid Plug-in': 375, 'Superior': 971, 'Series 62': 900, 'Nomad': 694, '9-5 SportCombi': 83, 'Plaza': 727, 'SVX': 882, 'Spitfire': 952, 'Omni': 697, 'Hummer': 520, 'Z3 M': 1105, 'Spider': 951, 'Elite': 354, 'Jetstar 88': 543, 'TD': 975, 'Sprite': 957, 'Ventura': 1045, 'Customline': 299, 'Ninety-Eight': 689, 'Falcon': 420, 'Series 75': 901, 'Fury': 441, 'Stylemaster': 962, 'R/V 20': 765, 'Can Am': 227, 'Super': 965, 'Rambler American': 810, '1100': 1, 'Gran Sport': 496, 'Skylark': 936, 'TC': 974, 'P1800': 707, 'Vanagon': 1039, 'Vista Cruiser': 1057, 'Master': 643, 'Silver Seraph': 919, '230SL': 9, 'Brooklands R': 172, 'GTX': 479, 'AMX': 107, 'Esprit': 377, 'Cutlass Calais': 301, 'Hornet': 518, '348': 26, 'Chieftain': 247, 'Super Bee': 966, 'RAM 250': 771, 'Discovery Series II': 318, 'Crossfire SRT-6': 288, 'Montero': 665, 'Q45': 743, 'F-1': 390, 'Coronet': 277, 'Reno': 830, 'Mark IX': 638, 'M35h': 601, 'MAZDASPEED6': 613, 'FX50': 418, 'Q5 Hybrid': 745, 'Eighty-Eight': 342, 'Milan Hybrid': 650, 'Ranger Chassis': 820, 'Silvia': 934, 'IPL G': 528, 'CJ-7': 192, 'Lumina Minivan': 595, 'Shadow': 904, 'Pajero': 711, 'i-Series': 1115, 'Aura Hybrid Green Line': 139, '626': 67, 'Ascender': 130, 'V40': 1030, 'Sprinter Passenger': 956, 'Grand Voyager': 506, 'S-TYPE R': 852, 'Eighty-Eight Royale': 343, '600': 64, 'Cordoba': 272, 'Cutlass': 300, 'MR2': 626, 'Intrepid': 538, 'Silverado SS': 933, 'Typhoon': 1023, 'Villager': 1055, 'RAM Wagon': 778, 'Syclone': 973, 'Z8': 1108, 'FX45': 417, 'GS 430': 469, 'Civic del Sol': 254, 'CT6 Hybrid Plug-In ': 205, 'Silver Shadow II': 920, 'Millenia': 651, 'S60 R': 862, 'Special Service Hybrid Plug-In ': 949, 'GS 200t': 465, 'Azzurra': 150, 'Road Runner': 837, 'Duster': 321, 'Catalina': 235, 'Trans Sport': 1009, 'LTD': 575, 'Belvedere': 159, 'Saratoga': 890, 'Tiger': 995, 'GS 400': 468, 'Sierra Classic 3500': 917, '2002': 8}


st.markdown("#")
st.subheader(":blue[Input Car Features]")



col2, col3 = st.columns(2)
with col2:
    model_name = model_name_map[
        col2.selectbox("Model Name", list(model_name_map.keys()))
    ]
    mileage = col2.number_input("Mileage", min_value=0.0, max_value=4290461.0, step=1.0)
    horsepower = col2.slider("Horsepower", 70.0, 903.0, 1.0)
    fuel_tank_volume = col2.slider("Fuel Tank Volume", 1.9, 64.0, 1.0)
    height = col2.slider("Height", 0, 603, 1)
    width = col2.slider("Width", 42.5, 109.0, 1.0)

with col3:
    make_name = make_name_map[col3.selectbox("Make Name", list(make_name_map.keys()))]
    franchise_make = franchise_make_map[col3.selectbox("Franchise Make", list(franchise_make_map.keys()))]
    daysonmarket = col3.slider("Days On Market", 0, 2716, 1)
    wheelbase = col3.slider("Wheel Base", 73.5, 201.0, 1.0)
    power = col3.slider("Power", 70.0, 903.0, 1.0)
    engine_displacement = col3.slider("Engine Displacement", 700.0, 8400.0, 1.0)
    # wheel_system = wheel_system_map[
    #     col3.selectbox("Wheel System", list(wheel_system_map.keys()))
    # ]


st.markdown("#")
col4, col5 = st.columns([3, 4])

with col4:
    pass
with col5:
    predict = st.button("Predict")


 #----------------Dataset Section---------------#
input_data = {
    "horsepower": horsepower,
    "mileage": mileage,
    "fuel_tank_volume": fuel_tank_volume,
    "make_name": make_name,
    "franchise_make": franchise_make,
    "model_name":model_name,
    "height": height,
    "power":power,
    "wheelbase": wheelbase,
    "daysonmarket": daysonmarket,
    "engine_displacement":engine_displacement,
    "width":width
    
}


#     # Create input data
input_features = pd.DataFrame(input_data, index=[0])


# -----------------Prediction-----------------------
st.markdown("#")
st.subheader("**:blue[Model's Prediction]**")


try:
    if predict and selected_model == "Decision Tree":
        scaled_features = st.session_state.scaler.transform(input_features)
        predicted_price = st.session_state.DT_model.predict(scaled_features)
        st.markdown(
                f"**The :red[{selected_model}] model estimates that the car is worth :red[💲{round(predicted_price[0], 2)}]**"
            )
    elif predict and selected_model == "Gradient Boosting":
        scaled_features = st.session_state.scaler.transform(input_features)
        predicted_price = st.session_state.GBR_model.predict(scaled_features)
        st.markdown(
                f"**The :red[{selected_model}] model estimates that the car is worth :red[💲{round(predicted_price[0], 2)}]**"
            )  
    else:
        st.empty()
except AttributeError:
    st.error("Please ensure to load the model from the sidebar⚠️")
