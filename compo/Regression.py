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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os


def Regression(df, target):
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
        "<h3 class='miniTitle' style ='margin-left:400px; margin-top:-40px;color: #00A8A8;'>Machine Learning-Regression</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:30px;">Sélectionnez l'algorithme souhaité</p>""",
        unsafe_allow_html=True,
    )
    algo = st.sidebar.selectbox(
        "",
        options=[
            "Default",
            "Regression Linéaire",
            "Regression polynomial",
            "SVM",
            "Decision Tree",
            "Random Forest",
        ],
    )

    # Vérification des données
    if target not in df.columns:
        st.error(f"La colonne cible '{target}' n'existe pas dans le DataFrame.")
        return

    # Gestion des valeurs nulles
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:  # Variable continue
                df[col].fillna(df[col].mean(), inplace=True)
            else:  # Variable catégorielle
                df[col].fillna(df[col].mode()[0], inplace=True)
    # Séparation des caractéristiques et de la cible
    X = df.drop(columns=[target])
    y = df[target]
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

    # Encodage de la cible si non numérique
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)

    if algo == "Regression Linéaire":
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Créer le modèle de régression linéaire
        model = LinearRegression()

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédire sur les données de test
        y_pred = model.predict(X_test)

        # Calcul des performances du modèle
        # Calcul des performances
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Affichage des résultats
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Mean Absolute Error (MAE) :  <span style="color: #6ec3c2;">{mae}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Erreur quadratique moyenne (MSE) :  <span style="color: #6ec3c2;">{mse}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Coefficient de détermination (R2) :  <span style="color: #6ec3c2;">{r2}.</p>""",
            unsafe_allow_html=True,
        )

        # Visualisation des prédictions par rapport aux valeurs réelles
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Valeurs Réelles vs Prédictions
            </h3>""",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(
            figsize=(4, 2.5)
        )  # Réduire davantage la taille de la figure
        ax.scatter(y_test, y_pred, color="blue", s=10)  # Diminuer la taille des points
        ax.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            color="red",
            linestyle="--",
            linewidth=1,
        )
        ax.set_xlabel("Valeurs réelles", fontsize=8)  # Réduire la taille du label
        ax.set_ylabel("Prédictions", fontsize=8)  # Réduire la taille du label
        ax.set_title(
            "Valeurs Réelles vs Prédictions", fontsize=9
        )  # Réduire la taille du titre
        st.pyplot(fig)

        # Visualisation des résidus
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Graphique : Résidus
            </h3>""",
            unsafe_allow_html=True,
        )
        residuals = y_test - y_pred
        fig_residuals = plt.figure(
            figsize=(4, 2.5)
        )  # Réduire davantage la taille de la figure
        sns.histplot(residuals, kde=True, color="blue", bins=20)
        plt.title("Distribution des Résidus", fontsize=9)  # Réduire la taille du titre
        plt.xlabel("Résidus", fontsize=7)  # Réduire la taille des labels
        plt.ylabel("Fréquence", fontsize=7)  # Réduire la taille des labels
        st.pyplot(fig_residuals)

        # Visualisation de la matrice de corrélation pour les variables explicatives
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Corrélation
            </h3>""",
            unsafe_allow_html=True,
        )
        corr_matrix = X.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            labels=dict(
                x="Caractéristiques", y="Caractéristiques", color="Corrélation"
            ),
            color_continuous_scale="Blues",
            title=" ",
            height=800,
            width=800,
        )
        fig_corr.update_layout(title_x=0.5)
        st.plotly_chart(fig_corr)
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
        report_filename = st.text_input("", value="rapport_classification.csv")
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour les coefficients</p>""",
            unsafe_allow_html=True,
        )
        importance_filename = st.text_input("", value="coefficients_importance.csv")
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour le modèle</p>""",
            unsafe_allow_html=True,
        )
        model_filename = st.text_input("", value="model_logistic_regression.pkl")

        if st.button("Exporter les fichiers"):
            # Exporter les résultats
            report_df = pd.DataFrame({"MAE": [mae], "MSE": [mse], "R2": [r2]})
            report_df.to_csv(report_filename, index=False)
            importance = pd.DataFrame({"Importance": model.coef_[0]})
            importance.to_csv(importance_filename, index=False)

            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "Regression polynomial":
        # Choix du degré du polynôme
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Choisissez le degré du polynôme</p>""",
            unsafe_allow_html=True,
        )
        degree = st.sidebar.slider("", min_value=1, max_value=5, value=2)

        # Création du pipeline avec régression polynomiale
        model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=degree)),
                ("regressor", LinearRegression()),
            ]
        )

        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Calcul des performances
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Affichage des résultats
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Mean Absolute Error (MAE) :  <span style="color: #6ec3c2;">{mae}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Erreur quadratique moyenne (MSE) :  <span style="color: #6ec3c2;">{mse}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Coefficient de détermination (R2) :  <span style="color: #6ec3c2;">{r2}.</p>""",
            unsafe_allow_html=True,
        )

        # Visualisation des prédictions par rapport aux valeurs réelles
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Graphique : Valeurs Réelles vs Prédictions
            </h3>""",
            unsafe_allow_html=True,
        )

        # Visualisation de la régression
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, ax=ax, line_kws={"color": "red"})
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        ax.set_title("Régression Polynomiale : Réel vs Prédictions")

        st.pyplot(fig)
        # Visualisation de la matrice de corrélation pour les variables explicatives
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Corrélation
            </h3>""",
            unsafe_allow_html=True,
        )
        corr_matrix = X.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            labels=dict(
                x="Caractéristiques", y="Caractéristiques", color="Corrélation"
            ),
            color_continuous_scale="Blues",
            title=" ",
            height=800,
            width=800,
        )
        fig_corr.update_layout(title_x=0.5)
        st.plotly_chart(fig_corr)
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
        model_filename = st.text_input("", value="model_logistic_regression.pkl")

        if st.button("Exporter les fichiers"):
            # Exporter le rapport des résultats
            results = pd.DataFrame({"MAE": [mae], "MSE": [mse], "R2": [r2]})
            results.to_csv(report_filename, index=False)

            # Exporter les coefficients d'importance (par exemple pour un modèle de régression)
            # Adapter selon la structure de ton modèle
            coefficients = pd.DataFrame(
                model.named_steps["regressor"].coef_, columns=["Coefficients"]
            )
            coefficients.to_csv(importance_filename, index=False)

            # Sauvegarder le modèle
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "SVM":
        # Séparation en données d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        # Choix des hyperparamètres pour le modèle SVM
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Choisissez le C</p>""",
            unsafe_allow_html=True,
        )
        C = st.sidebar.slider("", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Choisissez le kernel</p>""",
            unsafe_allow_html=True,
        )
        kernel = st.sidebar.selectbox(
            "", options=["linear", "poly", "rbf", "sigmoid"], index=2
        )
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Choisissez le gamma</p>""",
            unsafe_allow_html=True,
        )
        gamma = st.sidebar.selectbox(
            "",
            options=["scale", "auto"]
            + [float(i) for i in np.arange(0.1, 2.1, 0.1).tolist()],
        )

        # Création du modèle SVM
        model = SVR(C=C, kernel=kernel, gamma=gamma)
        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calcul des performances
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Affichage des résultats
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Mean Absolute Error (MAE) :  <span style="color: #6ec3c2;">{mae}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Erreur quadratique moyenne (MSE) :  <span style="color: #6ec3c2;">{mse}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Coefficient de détermination (R2) :  <span style="color: #6ec3c2;">{r2}.</p>""",
            unsafe_allow_html=True,
        )
        # Visualisation des prédictions par rapport aux valeurs réelles
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Graphique : Valeurs Réelles vs Prédictions
            </h3>""",
            unsafe_allow_html=True,
        )

        # Visualisation de la régression
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, ax=ax, line_kws={"color": "red"})
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        ax.set_title("Régression Polynomiale : Réel vs Prédictions")

        st.pyplot(fig)
        # Visualisation de la matrice de corrélation pour les variables explicatives
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Corrélation
            </h3>""",
            unsafe_allow_html=True,
        )
        corr_matrix = X.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            labels=dict(
                x="Caractéristiques", y="Caractéristiques", color="Corrélation"
            ),
            color_continuous_scale="Blues",
            title=" ",
            height=800,
            width=800,
        )
        fig_corr.update_layout(title_x=0.5)
        st.plotly_chart(fig_corr)
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
        model_filename = st.text_input("", value="model_svm.pkl")

        if st.button("Exporter les fichiers"):
            # Export des résultats dans un fichier CSV
            report_df = pd.DataFrame(
                {
                    "True Values": y_test,
                    "Predictions": y_pred,
                    "MAE": mae,
                    "MSE": mse,
                    "R2": r2,
                }
            )
            report_df.to_csv(report_filename, index=False)
            # Export des coefficients (si applicable)
            importance = pd.DataFrame(model.coef_).T  # Exemple pour un modèle SVM
            importance.to_csv(importance_filename, index=False)
            # Export du modèle SVM
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "Decision Tree":
        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Profondeur maximale de l'arbre</p>""",
            unsafe_allow_html=True,
        )
        max_depth = st.sidebar.slider("", min_value=1, max_value=20, value=5)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Nombre minimal d'échantillons pour diviser un nœud</p>""",
            unsafe_allow_html=True,
        )
        min_samples_split = st.sidebar.slider("", min_value=2, max_value=20, value=2)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Nombre minimal d'échantillons par feuille</p>""",
            unsafe_allow_html=True,
        )
        min_samples_leaf = st.sidebar.slider("", min_value=1, max_value=20, value=1)

        # Créer le modèle de régression basé sur les paramètres choisis
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)
        # Calcul des performances
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Affichage des résultats
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Mean Absolute Error (MAE) :  <span style="color: #6ec3c2;">{mae}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Erreur quadratique moyenne (MSE) :  <span style="color: #6ec3c2;">{mse}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Coefficient de détermination (R2) :  <span style="color: #6ec3c2;">{r2}.</p>""",
            unsafe_allow_html=True,
        )
        # Visualisation des prédictions par rapport aux valeurs réelles
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Graphique : Valeurs Réelles vs Prédictions
            </h3>""",
            unsafe_allow_html=True,
        )

        # Visualisation de la régression
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, ax=ax, line_kws={"color": "red"})
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        ax.set_title("Régression Polynomiale : Réel vs Prédictions")

        st.pyplot(fig)
        # Visualisation de la matrice de corrélation pour les variables explicatives
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Corrélation
            </h3>""",
            unsafe_allow_html=True,
        )
        corr_matrix = X.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            labels=dict(
                x="Caractéristiques", y="Caractéristiques", color="Corrélation"
            ),
            color_continuous_scale="Blues",
            title=" ",
            height=800,
            width=800,
        )
        fig_corr.update_layout(title_x=0.5)
        st.plotly_chart(fig_corr)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Visualisation de l'arbre de décision
            </h3>""",
            unsafe_allow_html=True,
        )
        # Afficher l'arbre avec plot_tree
        from sklearn.tree import plot_tree

        fig_tree = plt.figure(figsize=(15, 10))
        plot_tree(model, filled=True, feature_names=X.columns, fontsize=10)
        st.pyplot(fig_tree)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Importance des caractéristiques
            </h3>""",
            unsafe_allow_html=True,
        )
        feature_importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        # Créer un graphique de l'importance des caractéristiques
        fig_importance = px.bar(
            feature_importance_df, x="Feature", y="Importance", title=""
        )
        st.plotly_chart(fig_importance)
        # Export des résultats
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Export des Résultats
            </h3>""",
            unsafe_allow_html=True,
        )
        report_filename = st.text_input(
            "Nom du fichier pour le rapport", "rapport_classification.csv"
        )
        importance_filename = st.text_input(
            "Nom du fichier pour les coefficients", "coefficients_importance.csv"
        )
        model_filename = st.text_input(
            "Nom du fichier pour le modèle", "model_decision_tree.pkl"
        )

        if st.button("Exporter les fichiers"):
            # Exporter les résultats
            report_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            report_df.to_csv(report_filename, index=False)

            # Exporter les importances des caractéristiques
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": feature_importances}
            )
            feature_importance_df.to_csv(importance_filename, index=False)

            # Exporter le modèle
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "Random Forest":
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Nombre d'arbres (n_estimators)</p>""",
            unsafe_allow_html=True,
        )
        n_estimators = st.sidebar.slider("", min_value=10, max_value=200, value=100)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Profondeur maximale des arbres (max_depth)</p>""",
            unsafe_allow_html=True,
        )
        max_depth = st.sidebar.slider("", min_value=1, max_value=20, value=10)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Nombre minimum d'échantillons pour diviser un noeud (min_samples_split)</p>""",
            unsafe_allow_html=True,
        )
        min_samples_split = st.sidebar.slider("", min_value=2, max_value=20, value=2)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Nombre minimum d'échantillons pour une feuille (min_samples_leaf)</p>""",
            unsafe_allow_html=True,
        )
        min_samples_leaf = st.sidebar.slider("", min_value=1, max_value=20, value=1)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 10px; margin-bottom: -35px;">Nombre maximum de caractéristiques à considérer (max_features)</p>""",
            unsafe_allow_html=True,
        )
        max_features = st.sidebar.selectbox("", options=["sqrt", "log2", None])

        # Créer le modèle Random Forest avec les hyperparamètres définis par l'utilisateur
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Faire des prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calcul des performances
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Affichage des résultats
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Mean Absolute Error (MAE) :  <span style="color: #6ec3c2;">{mae}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Erreur quadratique moyenne (MSE) :  <span style="color: #6ec3c2;">{mse}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Coefficient de détermination (R2) :  <span style="color: #6ec3c2;">{r2}.</p>""",
            unsafe_allow_html=True,
        )
        # Visualisation des prédictions par rapport aux valeurs réelles
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Graphique : Valeurs Réelles vs Prédictions
            </h3>""",
            unsafe_allow_html=True,
        )

        # Visualisation de la régression
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, ax=ax, line_kws={"color": "red"})
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        ax.set_title("Régression Polynomiale : Réel vs Prédictions")

        st.pyplot(fig)

        # Visualisation de l'importance des caractéristiques
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Importance des caractéristiques
            </h3>""",
            unsafe_allow_html=True,
        )
        feature_importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        # Créer un graphique de l'importance des caractéristiques
        fig_importance = px.bar(
            feature_importance_df, x="Feature", y="Importance", title=""
        )
        st.plotly_chart(fig_importance)

        # Afficher tous les arbres du RandomForest
        # **Visualisation de tous les arbres**
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Visualisation des Arbres dans le Random Forest
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-size: 20px; color: #f7dace; margin-bottom: -20px;">Sélectionnez l'indice de l'arbre à visualiser:</p>""",
            unsafe_allow_html=True,
        )
        tree_index = st.slider("", 0, n_estimators - 1, 0)

        from sklearn.tree import plot_tree

        fig_tree, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model.estimators_[tree_index],  # Sélectionner un arbre du Random Forest
            feature_names=X.columns,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
        )
        st.pyplot(fig_tree)
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
        model_filename = st.text_input("", value="model_random_forest.pkl")

        if st.button("Exporter les fichiers"):
            # Exporter les résultats sous forme de CSV
            report_df = pd.DataFrame({"Réel": y_test, "Prédiction": y_pred})
            report_df.to_csv(report_filename, index=False)

            # Exporter l'importance des caractéristiques sous forme de CSV
            importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": model.feature_importances_}
            )
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            importance_df.to_csv(importance_filename, index=False)

            # Exporter le modèle sous forme de fichier pickle
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

            # Option pour exporter les graphiques
            st.markdown(
                f"""<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour la visualisation des prédictions</p>""",
                unsafe_allow_html=True,
            )
            plot_filename = st.text_input("", value="visualisation_predictions.png")

            # Enregistrer la visualisation des prédictions
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=y_test, y=y_pred, ax=ax, line_kws={"color": "red"})
            ax.set_xlabel("Valeurs réelles")
            ax.set_ylabel("Prédictions")
            ax.set_title("Régression Polynomiale : Réel vs Prédictions")
            fig.savefig(plot_filename)
            st.success(
                f"Le graphique a été exporté avec succès sous le nom : {plot_filename}"
            )

            # Option pour exporter l'importance des caractéristiques
            importance_plot_filename = st.text_input(
                "", value="importance_features.png"
            )
            fig_importance = px.bar(
                importance_df, x="Feature", y="Importance", title=""
            )
            fig_importance.write_image(importance_plot_filename)
            st.success(
                f"Le graphique d'importance des caractéristiques a été exporté avec succès sous le nom : {importance_plot_filename}"
            )
    # Titre de l'interface
    st.markdown(
        """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Tester le Modèle
        </h3>""",
        unsafe_allow_html=True,
    )

    # Initialisation de df_test_global_regression dans st.session_state
    if "df_test_global_regression" not in st.session_state:
        st.session_state.df_test_global_regression = pd.DataFrame()

    # Vérification que les données d'entraînement existent
    if not X.empty:
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Sélectionnez les fonctionnalités à utiliser pour le test :</p>""",
            unsafe_allow_html=True,
        )

        # Sélection des fonctionnalités à inclure dans le test
        selected_columns = st.multiselect(
            "",
            options=X.columns.tolist(),
        )

        if selected_columns:
            # DataFrame global pour conserver les lignes précédentes
            if st.session_state.df_test_global_regression.empty:
                st.session_state.df_test_global_regression = pd.DataFrame(
                    columns=selected_columns
                )

            # Initialisation du dictionnaire temporaire avec les colonnes sélectionnées
            if "temp_row" not in st.session_state:
                st.session_state.temp_row = {column: "" for column in selected_columns}
            else:
                # Mettre à jour pour inclure uniquement les colonnes sélectionnées
                for column in selected_columns:
                    if column not in st.session_state.temp_row:
                        st.session_state.temp_row[column] = ""

            # Formulaire d'insertion des valeurs
            st.markdown(
                f"""<p style="font-size: 20px; color: #f7dace;">Entrez les valeurs pour chaque colonne :</p>""",
                unsafe_allow_html=True,
            )

            # Collecte des valeurs des colonnes sélectionnées
            for column in selected_columns:
                st.session_state.temp_row[column] = st.text_input(
                    f"Entrez une valeur pour {column} :",
                    st.session_state.temp_row[column],
                    key=f"test_input_{column}",
                )

            # Bouton pour valider l'ajout de la ligne
            if st.button("Ajouter la ligne"):
                # Vérification que toutes les colonnes ont une valeur
                if all(value != "" for value in st.session_state.temp_row.values()):
                    # Ajout de la ligne au DataFrame global
                    new_row = pd.DataFrame([st.session_state.temp_row])
                    st.session_state.df_test_global_regression = pd.concat(
                        [st.session_state.df_test_global_regression, new_row],
                        ignore_index=True,
                    )
                    # Réinitialisation du dictionnaire temporaire
                    st.session_state.temp_row = {
                        column: "" for column in selected_columns
                    }
                    st.success("Ligne ajoutée avec succès !")
                else:
                    st.error(
                        "Veuillez remplir toutes les colonnes avant d'ajouter une ligne."
                    )

            st.markdown(
                f"""<p style="font-size: 20px; color: #f7dace;">Aperçu des données de test :</p>""",
                unsafe_allow_html=True,
            )
            st.dataframe(st.session_state.df_test_global_regression)

            # Vérifier si des données sont présentes avant la prédiction
            if st.session_state.df_test_global_regression.empty:
                st.error("Aucune donnée test n'est disponible.")
            else:
                # Prédiction sur les données test
                if st.button("Prédire"):
                    df_test = st.session_state.df_test_global_regression.copy()

                    if not df_test.empty:
                        # Encodage des colonnes catégoriques dans le jeu de test
                        for col in non_numeric_cols:
                            if col in df_test.columns:
                                le = LabelEncoder()
                                le.fit(
                                    df[col].astype(str)
                                )  # Fit sur les données d'entraînement
                                df_test[col] = df_test[col].map(
                                    lambda x: (
                                        le.transform([x])[0] if x in le.classes_ else -1
                                    )
                                )

                        # Gestion des colonnes manquantes dans df_test
                        for col in X.columns:
                            if col not in df_test.columns:
                                df_test[col] = 0  # Remplir avec une valeur par défaut

                        # Alignement des colonnes
                        df_test = df_test[X.columns]

                        # Normalisation des données de test
                        df_test_scaled = scaler.transform(df_test)

                        # Réalisation de la prédiction
                        predictions = model.predict(df_test_scaled)

                        # Affichage des résultats
                        st.success("Les prédictions du modèle sont :")
                        for i, pred in enumerate(predictions, start=1):
                            st.write(f"Observation {i} : {pred}")
                    else:
                        st.error("Aucune donnée test n'est disponible.")
