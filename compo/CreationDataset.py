import streamlit as st
import pandas as pd
import os


def creer_dataset():
    st.markdown(
        "<h3 class='miniTitle' style ='margin-left:410px; margin-top:-20px;color: #00A8A8;'>Création des données !</h3>",
        unsafe_allow_html=True,
    )
    # Demander le nombre de variables (colonnes)
    st.markdown(
        """<p style="font-size: 17px; color: #f7dace; margin-bottom: -50px;">Combien de variables voulez-vous dans votre dataset ?</p>""",
        unsafe_allow_html=True,
    )
    n_variables = st.number_input("", min_value=1, max_value=10)

    # Initialiser une liste vide pour les noms des variables
    noms_variables = []

    # Demander les noms des variables
    for i in range(n_variables):
        st.markdown(
            f"""<p style="font-size: 17px; color: #f7dace; margin-bottom: -50px;">Nom de la variable {i+1}:</p>""",
            unsafe_allow_html=True,
        )
        nom_variable = st.text_input(f"", key=f"var_{i}")
        noms_variables.append(nom_variable)

    # Demander le nombre d'enregistrements (lignes)
    st.markdown(
        """<p style="font-size: 17px; color: #f7dace; margin-bottom: -50px;">Combien d'enregistrements voulez-vous dans votre dataset ?</p>""",
        unsafe_allow_html=True,
    )
    n_enregistrements = st.number_input("", min_value=1)

    # Initialiser une liste pour stocker les données des enregistrements
    data = {nom: [] for nom in noms_variables}

    # Créer un formulaire pour saisir les données des enregistrements
    for i in range(n_enregistrements):
        st.markdown(
            f"""<h3 style="color: #A96255;">Enregistrement {i+1}</h3>""",
            unsafe_allow_html=True,
        )
        for j, nom_variable in enumerate(noms_variables):
            # Ajouter un identifiant unique en combinant l'index de l'enregistrement, l'index de la variable et le nom de la variable
            st.markdown(
                f"""<p style="font-size: 17px; color: #f7dace; margin-bottom: -50px;">Entrez la valeur pour {nom_variable} (Enregistrement {i+1}):</p>""",
                unsafe_allow_html=True,
            )
            key = f"enregistrement_{i}_{j}_{nom_variable}"  # Clé unique
            valeur = st.text_input(f"", key=key)
            data[nom_variable].append(valeur)

    # Demander le nom du dataset
    st.markdown(
        """<h3 style="color: #A96255;">Nom du dataset</h3>""", unsafe_allow_html=True
    )
    st.markdown(
        """<p style="font-size: 17px; color: #f7dace; margin-bottom: -50px;">Entrez le nom du dataset (sans extension .csv) :</p>""",
        unsafe_allow_html=True,
    )
    nom_dataset = st.text_input("")
    csv = None
    # Vérifier si le bouton a été cliqué
    if "btn_clicked" not in st.session_state:
        st.session_state.btn_clicked = False

    # Ajouter un bouton personnalisé via HTML
    btn_html = """
        <style>/* From Uiverse.io by McHaXYT */ 
            .button {
            cursor: pointer;
            font-size: 1.5rem;  /* Augmenter la taille de la police */
            line-height: 2rem;  /* Ajuster la hauteur de ligne pour un meilleur espacement */
            padding: 0.7rem 1.7rem;  /* Augmenter les marges internes */
            color: #1f3861;
            background-color: rgb(79 70 229);
            background: linear-gradient(144deg, #e97f4f, #00A8A8 90%, #00ddeb);
            font-weight: 600;
            border-radius: 1rem;  /* Augmenter l'arrondi des bords */
            border-style: none;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.35s linear;
            text-align: center;  /* S'assurer que le texte est centré */
            width: auto;  /* Laisser le bouton s'ajuster à son contenu */
            max-width: 800px;  /* Ajouter une largeur maximale pour contrôler la taille */
            margin: 0 auto;  /* Centrer le bouton horizontalement */
        }

            .button:hover {
            box-shadow: inset 0 5px 25px 0 #e97f4f, inset 0 10px 15px 0px #5b42f3,
                inset 0 5px 25px 0px #00ddeb;
            }
        </style>
        <button class="button">
            <svg
                xmlns="http://www.w3.org/2000/svg"
                width="1.25rem"
                height="1.25rem"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
            >
                <path d="M12 19v-7m0 0V5m0 7H5m7 0h7"></path>
            </svg>
            Create
        </button>
        """

    # Gestion de l'état du bouton
    if st.markdown(
        f"""
    <script>
    const button = document.querySelector('.button');
    button.addEventListener('click', () => {{
        window.parent.postMessage({{isButtonClicked: true}}, "*");
    }});
    </script>
    """,
        unsafe_allow_html=True,
    ):
        st.session_state.btn_clicked = True

    # Si le bouton a été cliqué
    if st.session_state.btn_clicked:
        # Créer le DataFrame
        df = pd.DataFrame(data)

        # Afficher le dataset créé
        st.markdown(
            """<p style="font-size: 17px; color: #f7dace;">Votre dataset créé :</p>""",
            unsafe_allow_html=True,
        )
        st.write(df.head())

        # Sauvegarder le dataset sous forme de fichier CSV
        if nom_dataset:
            file_path = os.path.join(
                os.getcwd(), f"{nom_dataset}.csv"
            )  # Créer le chemin du fichier dans le répertoire courant
            csv = df.to_csv(index=False).encode("utf-8")

            # Sauvegarder le fichier CSV dans le même répertoire que l'application
            with open(file_path, "wb") as f:
                f.write(csv)

            # Afficher un message de succès
            st.success(f"Le dataset a été sauvegardé sous : {file_path}")

        # Fournir un bouton de téléchargement pour le fichier CSV
        if csv:
            st.download_button(
                label="Télécharger le fichier CSV",
                data=csv,
                file_name=f"{nom_dataset}.csv",
                mime="text/csv",
            )
    # Afficher le bouton personnalisé avec st.markdown
    st.markdown(btn_html, unsafe_allow_html=True)
