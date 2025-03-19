import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from compo.CreationDataset import creer_dataset
from compo.ImportDataset import ImportDataset
from compo.ExplorationData import ExplorationData
from compo.NettoyageDataset import NettoyageDataset
from compo.Classification import Classification
from compo.Regression import Regression
from compo.Clustering import Clustering

from compo.NeuralNetworks import NeuralNetworks
import os


st.set_page_config(page_title="SMARTAnalyzr", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@700&display=swap');

        .title {
            text-align: center;
            font-family: 'Raleway', sans-serif;
            background: linear-gradient(90deg, #e97f4f, #ffb36b);
            -webkit-background-clip: text;
            color: transparent;
            font-size: 50px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            margin-bottom: 50px;
            color : #e97f4f;
            margin-top: -50px;
        }

        .subtitle {
            text-align: center;
            font-family: 'Raleway', sans-serif;
            color: #6ec3c2;
            font-size: 20px;
            font-weight: 500;
            margin-top: -10px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }

        .miniTitle{
            color: #e97f4f; 
            padding-left : 110px; 
            padding-top:30px; 
            font-size: 35px;
        }

        .paragraph{
            font-family: 'Raleway', sans-serif; 
            font-size: 18px; 
            color: #ffffff; 
            line-height: 1.6; 
            margin: 20px 95px; 
            padding: 20px; 
            border-radius: 10px;
            background-color: hsla(201, 56%, 24%, 0.25);
            color: #f7dace;
        }

        .tabs .stTab {
            font-size: 18px;
            font-family: 'Raleway', sans-serif;
            font-weight: bold;
            padding: 10px 20px;
            color: #ffffff;
            background-color: #e97f4f;
            border-radius: 10px;
            margin: 5px;
            text-align: center;
        }

        .tabs .stTab.selected {
            background-color: #ffb36b;
            color: #1f3861;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='title' style = 'color : #e97f4f;'>SMARTAnalyzr</h1>",
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title="",
    options=["Accueil", "Traitement", "Aide"],
    icons=["house", "tools", "question-circle"],
    menu_icon="menu",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "5px", "background-color": "#f7dace"},
        "icon": {"color": "#6ec3c2", "font-size": "20px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "color": "#333333",
        },
        "nav-link-selected": {"background-color": "#e97f4f", "color": "#1f3861"},
    },
)

if selected == "Accueil":
    st.markdown(
        "<h3 class='miniTitle' style = 'color: #e97f4f; '>Bienvenue sur SMARTAnalyzr !</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class = 'paragraph'>
            Votre partenaire incontournable pour l'analyse des données et la prise de décision éclairée. 
            <br><br>
            Chez <strong>SMARTAnalyzr</strong>, nous comprenons que vos données sont précieuses. 
            Que vous soyez <span style = "color : #6ec3c2;">étudiant</span>, <span style = "color : #6ec3c2;">chercheur</span>, ou <span style = "color : #6ec3c2;">professionnel</span>, <br>notre application est pensée pour 
            simplifier chaque étape de votre analyse, tout en vous offrant des résultats fiables et exploitables.<br><br>
            Transformer vos données en insights aujourd'hui avec SMARTAnalyzr!
        </div>
    """,
        unsafe_allow_html=True,
    )

if selected == "Traitement":
    selected1 = option_menu(
        menu_title="",
        options=["Upload Data", "Exploration des données", "Modèle"],
        icons=["upload", "table", "gear"],
        menu_icon="menu",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "#6ec3c2", "font-size": "20px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "center",
                "margin": "0px",
                "color": "#f7dace",
            },
            "nav-link-selected": {"background-color": "#e97f4f", "color": "#1f3861"},
        },
    )

    if selected1 == "Upload Data":
        st.sidebar.markdown(
            """
            <style>
                /* Supprime le padding et la marge au-dessus de l'image */
                .css-1v3n9a2, .css-1d391kg, .stSidebar {
                    padding-top: 0 !important;
                    margin-top: -80px !important;
                }

                /* Contrôle la marge de l'image dans la barre latérale */
                .sidebar-image {
                    width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 !important;  /* Enlève toute marge */
                    padding: 0 !important;  /* Enlève tout padding */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        logo_path = "assets/SMARTAnalyzr_Logo_Final.png"  # Assurez-vous que le chemin est correct
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, use_container_width=True, caption="")
        else:
            st.sidebar.warning("Logo non trouvé. Veuillez vérifier le chemin.")

        choix1, choix2 = st.tabs(["Importation du Dataset", "Création du Dataset"])

        with choix1:
            df = ImportDataset()
            st.session_state.df = df
        with choix2:
            creer_dataset()
    # Exploration des données
    if selected1 == "Exploration des données":
        if "df" in st.session_state and st.session_state.df is not None:
            # Vérifie si st.session_state.df a deux ou trois éléments
            if len(st.session_state.df) == 2:
                df, target = st.session_state.df
                type = None  # Initialise `type` si absent
            elif len(st.session_state.df) == 3:
                df, target, type = st.session_state.df
            else:
                st.error(
                    "Format inattendu pour `st.session_state.df`. Assurez-vous qu'il contient (df, target, type)."
                )
                st.stop()

            selected2 = option_menu(
                menu_title="",
                options=["Information générales", "Nettoyage des données"],
                icons=["info-circle", "brush"],
                menu_icon="menu",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {
                        "padding": "-5px",
                        "background-color": "rgba(255, 255, 255, 0)",
                    },
                    "icon": {"color": "#6ec3c2", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "17px",
                        "text-align": "center",
                        "margin": "-15px",
                        "color": "#f7dace",
                        "text-decoration": "underline",
                    },
                    "nav-link-selected": {
                        "background-color": "#0e1117",
                        "color": "#e97f4f",
                        "font-weight": "bold",
                    },
                },
            )

            if selected2 == "Information générales":
                ExplorationData(df_copy=df, target=target)
            elif selected2 == "Nettoyage des données":
                type = NettoyageDataset(df=df, target=target)
                st.session_state.df = (df, target, type)
                file_path = os.path.join(os.getcwd(), "dataset_nettoye.csv")
                df.to_csv(
                    file_path, index=False
                )  # Sauvegarder le fichier CSV dans le répertoire courant

                # Bouton pour télécharger le dataset nettoyé
                st.download_button(
                    label="Télécharger le dataset nettoyé",
                    data=open(
                        file_path, "rb"
                    ).read(),  # Lire le fichier en mode binaire
                    file_name="dataset_nettoye.csv",
                    mime="text/csv",
                )
        else:
            st.warning(
                "Aucun dataset chargé. Veuillez importer ou créer un dataset dans la section 'Upload Data'."
            )

    # Modèle
    if selected1 == "Modèle":
        if "df" in st.session_state and st.session_state.df is not None:
            # Vérifie si st.session_state.df contient les bonnes données (df, target, type)
            if len(st.session_state.df) == 2:
                df, target = st.session_state.df
                type = "Machine Learning"  # Définit type à None si non présent
            elif len(st.session_state.df) == 3:
                df, target, type = st.session_state.df
            else:
                st.error(
                    "Format inattendu pour `st.session_state.df`. Assurez-vous qu'il contient (df, target, type)."
                )
                st.stop()
            # Si 'type' est "Machine Learning"
            if type == "Machine Learning":
                if target is not None:
                    if df[target].nunique() <= 10 or df[target].dtype in [
                        "object",
                        "category",
                    ]:
                        Classification(df=df, target=target)
                    else:
                        Regression(df=df, target=target)
                else:
                    # Si target est None, on appelle Clustering
                    Clustering(df=df)

            # Si 'type' est "Deep Learning"
            elif type == "Deep Learning":
                if target is not None:
                    NeuralNetworks(df=df, target=target)
                else:
                    st.warning(
                        "Le target n'est pas spécifié. Impossible d'utiliser Neural Networks."
                    )
        else:
            st.warning(
                "Aucun dataset chargé. Veuillez importer ou créer un dataset dans la section 'Upload Data'."
            )
if selected == "Aide":
    st.markdown(
        "<h3 class='miniTitle' style ='margin-left:630px; color: #00A8A8;'>Aide!</h3>",
        unsafe_allow_html=True,
    )
    st.video("Aide.mp4")
