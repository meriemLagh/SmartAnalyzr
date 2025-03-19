import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import plotly.express as px
import os


def ExplorationData(df_copy, target):

    df = df_copy.copy()
    st.markdown(
        """<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">""",
        unsafe_allow_html=True,
    )

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

    logo_path = (
        "assets/SMARTAnalyzr_Logo_Final.png"  # Assurez-vous que le chemin est correct
    )
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True, caption="")
    else:
        st.sidebar.warning("Logo non trouvé. Veuillez vérifier le chemin.")
    st.sidebar.markdown(
        """<h2 style="color: #e97f4f; margin-bottom: -40px; text-align : center; font-size: 24px;">Résumé des étapes</h2>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<p style="font-size: 20px; color: #f7dace; margin-top : 15px;">I - Affichage<br>
        II - Taille du dataset <br>III - Statistiques déscriptives <br>IV- Types de données <br>V- Valeurs manquantes<br>VI - Vérification des doublons<br>VII - Sélection du variable<br>
        VIII - Visualisation</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 class='miniTitle' style ='margin-left:410px; margin-top:-40px;color: #00A8A8;'>Informations générales!</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<h3 style="color: #A96255; margin-top:30px;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Affichage
        </h3>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<p style="font-size: 20px; color: #f7dace; margin-bottom: -40px;">Affichage des 10 premiers lignes du dataset</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<style>
        .center-table {
        display: flex;
        justify-content: center;
        align-items: center;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown(
        df.head(10).to_html(classes="dataframe", index=False), unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Taille du dataset
        </h3>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -40px;">Le dataset contient <span style="color: #6ec3c2;">{df.shape[0]} lignes</span> et <span style="color: #6ec3c2;">{df.shape[1]} colonnes</span>.</p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px; margin-top:40px;"></i>
        Statistiques déscriptives
        </h3>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Statistiques de base pour les colonnes numériques :</p>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown(
        df.describe().to_html(classes="dataframe", index=False), unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Afficher les types de données de chaque colonne
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Types de données
        </h3>""",
        unsafe_allow_html=True,
    )
    types = df.dtypes
    types = types.reset_index()
    types.columns = ["Variable", "Type"]
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown(types.to_html(classes="dataframe", index=False), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Vérifier les valeurs manquantes
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Valeurs manquantes
        </h3>""",
        unsafe_allow_html=True,
    )
    som = df.isnull().sum()
    som = som.reset_index()  # Convertir la Série en DataFrame
    som.columns = ["Variable", "Valeurs"]  # Renommer les colonnes
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown(som.to_html(classes="dataframe", index=False), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Vérifier les doublons
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Vérification des doublons
        </h3>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Nombre de doublons dans le dataset: {df.duplicated().sum()}</p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px; margin-top:40px;"></i>
        Sélection du variable
        </h3>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Sélectionnez la variable sur laquelle vous souhaitez effectuer les études :</p>""",
        unsafe_allow_html=True,
    )

    variable = st.selectbox("", options=df.columns)

    st.sidebar.markdown(
        f"""<p style="font-size: 20px; color:rgb(239, 232, 229); margin-top : -15px; margin-left:10px;">1 - Histogramme de {variable}<br>
        2 - Répartition des classes de {variable}""",
        unsafe_allow_html=True,
    )
    if target is not None:
        st.sidebar.markdown(
            f"""<p style="font-size: 20px; color:rgb(239, 232, 229); margin-top : -15px; margin-left:10px;">3 - Distribution des classes cibles {target}<br>
            4 - Répartition des classes cibles {target}<br>5 - Distribution de {variable} par classe<br>6 - Nuage de points pour les caractéristiques<br>7 - Matrice de corrélation""",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f"""<p style="font-size: 20px; color:rgb(239, 232, 229); margin-top : -15px; margin-left:10px;">3 - Distribution de {variable} par classe<br>4 - Nuage de points pour les caractéristiques<br>5 - Matrice de corrélation""",
            unsafe_allow_html=True,
        )

    if is_numeric_dtype(df[variable]):
        df[variable] = pd.cut(
            df[variable],
            bins=3,  # Nombre d'intervalles
            labels=["Bas", "Moyen", "Haut"],  # Labels pour chaque intervalle
            include_lowest=True,
        )

    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px; margin-top:40px;"></i>
        Visualisation
        </h3>""",
        unsafe_allow_html=True,
    )

    # Définir une palette pastel
    pastel_colors = px.colors.qualitative.Pastel  # Palette pastel de Plotly

    # Histogramme
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Histogramme de {variable}</p>""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        px.histogram(df, x=variable, title="", color_discrete_sequence=pastel_colors)
    )
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Répartition des classes de {variable}</p>""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        px.pie(df, names=df[variable], title="", color_discrete_sequence=pastel_colors)
    )

    if target is not None:
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Distribution des classes cibles {target}</p>""",
            unsafe_allow_html=True,
        )
        # Bar chart
        st.plotly_chart(
            px.bar(
                df,
                x=df[variable],
                y=target,
                color=df[variable],
                title="",
                color_discrete_sequence=pastel_colors,
            )
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Répartition des classes cibles {target}</p>""",
            unsafe_allow_html=True,
        )
        # Pie chart
        st.plotly_chart(
            px.pie(
                df, names=df[target], title=" ", color_discrete_sequence=pastel_colors
            )
        )

    # Histogramme de la caractéristique sélectionnée par classe
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Distribution de {variable} par classe</p>""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        px.histogram(
            df,
            x=variable,
            color=df[variable],
            title=f"",
            color_discrete_sequence=pastel_colors,
        )
    )

    # Scatter matrix
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Nuage de points pour les caractéristiques</p>""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        px.scatter_matrix(
            df,
            dimensions=df.columns[:-1],
            color=df[variable],
            title="",
            color_discrete_sequence=pastel_colors,
        )
    )

    from sklearn.preprocessing import LabelEncoder

    # Convertir les colonnes catégorielles en numériques
    label_encoder = LabelEncoder()

    # Appliquer l'encodage à chaque colonne catégorielle
    for column in df.select_dtypes(include=["object", "category"]).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Calculer la matrice de corrélation
    correlation_matrix = df.corr()

    # Afficher la matrice de corrélation
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Matrice de corrélation</p>""",
        unsafe_allow_html=True,
    )
    fig = px.imshow(
        correlation_matrix,
        title="",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(x="Caractéristiques", y="Caractéristiques"),
    )

    # Ajuster la taille de la figure
    fig.update_layout(
        height=800,  # Hauteur de l'image (modifiez selon vos besoins)
        width=800,  # Largeur de l'image (modifiez selon vos besoins)
    )
    # Afficher dans Streamlit
    st.plotly_chart(fig)
