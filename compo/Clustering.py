import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    LabelEncoder,
    MinMaxScaler,
)
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from sklearn import metrics
import os


def Clustering(df):
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
        """<h2 style="color: #e97f4f; margin-bottom: -40px; text-align : center; font-size: 24px;">Les options disponibles</h2>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 class='miniTitle' style ='margin-left:400px; margin-top:-40px;color: #00A8A8;'>Machine Learning-Clustering</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<h3 style=" color: #f7dace; margin-top : -20px; margin-top:30px;">K-Means</h3>""",
        unsafe_allow_html=True,
    )

    # Gestion des valeurs nulles
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:  # Variable continue
                df[col].fillna(df[col].mean(), inplace=True)
            else:  # Variable catégorielle
                df[col].fillna(df[col].mode()[0], inplace=True)
    # Séparation des caractéristiques et de la cible
    X = df.copy()
    # Encodage des colonnes non numériques dans X
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        for col in non_numeric_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Normalisation des colonnes numériques
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    numeric_cols = X.select_dtypes(include=["number"]).columns
    X = X[numeric_cols]

    # Interface utilisateur pour choisir les paramètres
    n_clusters = st.sidebar.slider(
        "Nombre de clusters", min_value=2, max_value=10, value=3
    )
    init_method = st.sidebar.selectbox(
        "Méthode d'initialisation", options=["k-means++", "random"]
    )
    max_iter = st.sidebar.slider(
        "Nombre maximum d'itérations", min_value=100, max_value=1000, value=300
    )
    n_init = st.sidebar.slider(
        "Nombre de tentatives", min_value=1, max_value=20, value=10
    )

    # Création du modèle KMeans
    kmeans = KMeans(
        n_clusters=n_clusters, init=init_method, max_iter=max_iter, n_init=n_init
    )

    # Entraînement du modèle
    kmeans.fit(X)

    # Prédictions des clusters
    df["Cluster"] = kmeans.predict(X)

    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Performance du modèle
        </h3>""",
        unsafe_allow_html=True,
    )
    # Calcul du score de silhouette
    silhouette_avg = silhouette_score(X, df["Cluster"])
    calinski_harabasz = metrics.calinski_harabasz_score(X, df["Cluster"])
    # Calcul du score Davies-Bouldin
    davies_bouldin = metrics.davies_bouldin_score(X, df["Cluster"])
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace;">Score de Silhouette : <span style="color: #6ec3c2;">{silhouette_avg:.2f}</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace;">Score de Calinski Harabasz : <span style="color: #6ec3c2;">{calinski_harabasz:.2f}</p>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace;">Score de Davies Bouldin : <span style="color: #6ec3c2;">{davies_bouldin:.2f}</p>""",
        unsafe_allow_html=True,
    )

    # Visualisation 1 : Clustering en 2D (si possible)
    # Visualisation avec Plotly
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Clustering K-means avec centroïdes
        </h3>""",
        unsafe_allow_html=True,
    )
    fig = px.scatter(
        X,
        x=X.columns[0],
        y=X.columns[1],
        color=df["Cluster"],
        title="",
        labels={X.columns[0]: "Feature 1", X.columns[1]: "Feature 2"},
    )

    # Ajouter les centroïdes au graphique
    centroids = kmeans.cluster_centers_
    centroid_df = pd.DataFrame(centroids, columns=X.columns)
    fig.add_scatter(
        x=centroid_df[X.columns[0]],
        y=centroid_df[X.columns[1]],
        mode="markers",
        marker=dict(color="red", size=12, symbol="x"),
        name="Centroids",
    )

    # Affichage du graphique
    st.plotly_chart(fig)
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Nombre d'éléments par cluster
        </h3>""",
        unsafe_allow_html=True,
    )
    # Créer un graphique à barres pour afficher la distribution des clusters
    cluster_counts = df["Cluster"].value_counts()

    # Créer un graphique avec Plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=cluster_counts.index,  # Les indices (numéros des clusters)
                y=cluster_counts.values,  # Les valeurs (nombre d'éléments dans chaque cluster)
                marker_color="skyblue",
            )
        ]
    )

    # Ajouter des titres et labels
    fig.update_layout(
        title="",
        xaxis_title="Cluster",
        yaxis_title="Nombre d'éléments",
        template="plotly_white",
    )

    # Afficher le graphique avec Streamlit
    st.plotly_chart(fig)

    # Affichage des centres des clusters
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Centres des clusters
        </h3>""",
        unsafe_allow_html=True,
    )
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown(
        centers_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    # Export des résultats
    st.markdown(
        """<h3 style="color: #A96255;">
        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
        Export des Résultats
        </h3>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour le rapport de classification</p>""",
        unsafe_allow_html=True,
    )
    report_filename = st.text_input(
        "",
        value="rapport_classification.csv",
    )
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour les coefficients</p>""",
        unsafe_allow_html=True,
    )
    importance_filename = st.text_input("", value="coefficients_importance.csv")
    st.markdown(
        f"""<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour le modèle</p>""",
        unsafe_allow_html=True,
    )
    model_filename = st.text_input("", value="model_kmeans.pkl")

    if st.button("Exporter les fichiers"):
        # Exporter les résultats
        df.to_csv(report_filename, index=False)
        # Exporter les coefficients (centroïdes)
        centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
        centroids_df.to_csv(importance_filename, index=False)

        # Exporter le modèle
        with open(model_filename, "wb") as f:
            pickle.dump(kmeans, f)

        st.success(
            f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
        )
