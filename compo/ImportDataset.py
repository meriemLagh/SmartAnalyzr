import streamlit as st
import pandas as pd


def ImportDataset():
    st.markdown(
        "<h3 class='miniTitle' style ='margin-left:410px; margin-top:-20px;color: #00A8A8;'>Importation des données !</h3>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """<h4 style="color: #f7dace; margin-bottom: -40px;">Veuillez télécharger votre dataset de type <span style="color: #6ec3c2;">"csv"</span> souhaité à traiter ici</h4>""",
        unsafe_allow_html=True,
    )
    uploaded_file = st.sidebar.file_uploader("", type=["csv"])
    st.sidebar.markdown(
        """<h4 style="color: #f7dace; margin-bottom: -40px;"><span style="text-align : center;">Ou bien</span><br> Veuillez entrer l'URL du fichier à importer</h4>""",
        unsafe_allow_html=True,
    )
    dataset_url = st.sidebar.text_input(" ", "")

    df = None  # Définir df à None au départ

    if uploaded_file is not None:
        # Si un fichier est téléchargé localement
        df = pd.read_csv(uploaded_file)
    elif dataset_url:
        try:
            if dataset_url.endswith(".csv"):
                df = pd.read_csv(dataset_url)
            elif dataset_url.endswith(".txt"):
                # Lire un fichier texte en ligne
                df = pd.read_csv(
                    dataset_url, delimiter="\t"
                )  # Vous pouvez aussi essayer d'autres délimiteurs
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du fichier depuis l'URL: {e}")

    if df is not None:
        st.markdown(
            """
                    <style>
                        .center-table {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            margin-top: 20px;
                        }
                    </style>
                """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """<p style="font-size: 20px; color: #f7dace;">Voici l'affichage des 5 premiers enregistrements du dataset téléchargé</p>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            df.head().to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        existance_cible = st.sidebar.checkbox(
            "Cocher si vous avez une variable cible dans votre dataset"
        )

        if existance_cible:
            st.markdown(
                """<h3 style="color: #f7dace; margin-bottom: -40px;">Sélectionnez la <span style="color: #6ec3c2;">Variable Cible</span></h3>""",
                unsafe_allow_html=True,
            )
            target_variable = st.selectbox("", options=df.columns)
            st.markdown(
                f"""<p style="font-size: 20px; color: #f7dace;">Votre dataset sera traité par des modèles d'apprentissage automatique supervisé tel que la variable cible est : <strong>{target_variable}</strong></p>""",
                unsafe_allow_html=True,
            )
            return df, target_variable
        else:
            st.markdown(
                """<p style="font-size: 20px; color: #f7dace;">Votre dataset sera traité par des modèles d'apprentissage automatique non supervisé</p>""",
                unsafe_allow_html=True,
            )
        return df, None
