import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.figure_factory as ff
import pickle
import os


def Classification(df, target):
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
        "<h3 class='miniTitle' style ='margin-left:400px; margin-top:-40px;color: #00A8A8;'>Machine Learning-Classification</h3>",
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
            "Regression Logistique",
            "SVM",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "KNN",
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

    if algo == "Regression Logistique":
        # Détection du cas binaire ou multiclasse
        is_binary = len(np.unique(y)) == 2
        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Création du modèle de régression logistique
        model = LogisticRegression(
            multi_class="ovr" if is_binary else "multinomial",
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(X_train, y_train)

        # Prédiction sur les données de test
        y_pred = model.predict(X_test)

        # Évaluation des performances
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Accuracy :  <span style="color: #6ec3c2;">{acc:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Precision :  <span style="color: #6ec3c2;">{precision:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Recall :   <span style="color: #6ec3c2;">{recall:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">F1-score : <span style="color: #6ec3c2;">{f1:.2f}.</p>""",
            unsafe_allow_html=True,
        )

        # Rapport de classification en tableau
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.rename(columns={"index": "Classe"}, inplace=True)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Rapport de Classification
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            report_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Confusion
            </h3>""",
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        # Visualisation de la matrice de confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédictions", y="Véritables", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            height=600,  # Hauteur de l'image
            width=600,
        )
        fig.update_layout(title_text="", title_x=0.5)
        st.plotly_chart(fig)

        # Importance des coefficients
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Importance des caractéristiques
            </h3>""",
            unsafe_allow_html=True,
        )
        if is_binary:
            coefficients = model.coef_[0]
        else:
            coefficients = np.mean(model.coef_, axis=0)

        importance = pd.DataFrame(
            {"Caractéristique": X.columns, "Coefficient": coefficients}
        )
        importance = importance.sort_values(by="Coefficient", key=abs, ascending=False)

        fig_importance = px.bar(
            importance, x="Coefficient", y="Caractéristique", orientation="h", title=""
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
            report_df.to_csv(report_filename, index=False)
            importance.to_csv(importance_filename, index=False)

            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "SVM":
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:10px;">Sélectionnez le kernel souhaité</p>""",
            unsafe_allow_html=True,
        )
        kernel_choice = st.sidebar.selectbox("", ["linear", "rbf", "poly", "sigmoid"])
        # 5. Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 6. Création et entraînement du modèle SVM
        model = SVC(kernel=kernel_choice, random_state=42)
        model.fit(X_train, y_train)

        # 7. Évaluation du modèle
        y_pred = model.predict(X_test)
        # Évaluation des performances
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Accuracy :  <span style="color: #6ec3c2;">{acc:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Precision :  <span style="color: #6ec3c2;">{precision:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Recall :   <span style="color: #6ec3c2;">{recall:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">F1-score : <span style="color: #6ec3c2;">{f1:.2f}.</p>""",
            unsafe_allow_html=True,
        )

        # Rapport de classification en tableau
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.rename(columns={"index": "Classe"}, inplace=True)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Rapport de Classification
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            report_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Confusion
            </h3>""",
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        # Visualisation de la matrice de confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédictions", y="Véritables", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            height=600,  # Hauteur de l'image
            width=600,
        )
        fig.update_layout(title_text="", title_x=0.5)
        st.plotly_chart(fig)

        # Importance des coefficients
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Importance des caractéristiques
            </h3>""",
            unsafe_allow_html=True,
        )
        # Visualisation de l'hyperplan (2D)
        if X.shape[1] >= 2:  # Si les données ont plus de 1 dimension
            # Sélectionner les deux premières dimensions pour la visualisation de l'hyperplan
            X_vis = X_train.iloc[
                :, :2
            ].values  # On prend les deux premières dimensions pour la visualisation
            y_vis = y_train

            # Création de la grille pour afficher l'hyperplan
            h = 0.02  # Résolution de la grille
            x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
            y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            if X.shape[1] == 2:  # Cas de la classification binaire (2D)
                # Prédiction de l'hyperplan sur la grille avec le modèle formé sur les deux premières dimensions
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Visualisation du plan de décision
                fig = go.Figure()
                fig.add_trace(
                    go.Contour(
                        z=Z, x=xx[0], y=yy[:, 0], colorscale="RdBu_r", opacity=0.4
                    )
                )

                # Ajout des points d'entraînement
                fig.add_trace(
                    go.Scatter(
                        x=X_vis[:, 0],
                        y=X_vis[:, 1],
                        mode="markers",
                        marker=dict(color=y_vis, colorscale="Viridis", showscale=True),
                    )
                )

                # Configuration du graphique
                fig.update_layout(
                    title="Visualisation de l'Hypersplan SVM (Classification Binaire)",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    height=600,
                    width=600,
                )
                st.plotly_chart(fig)

            else:  # Cas de la classification multiclasse (plus de 2D)
                # Prédiction de l'hyperplan sur la grille avec le modèle formé sur toutes les dimensions
                Z = model.predict(
                    np.c_[
                        xx.ravel(),
                        yy.ravel(),
                        np.zeros_like(xx.ravel()),
                        np.zeros_like(xx.ravel()),
                    ]
                )  # Ajouter deux dimensions nulles
                Z = Z.reshape(xx.shape)

                # Visualisation du plan de décision
                fig = go.Figure()
                fig.add_trace(
                    go.Contour(
                        z=Z, x=xx[0], y=yy[:, 0], colorscale="RdBu_r", opacity=0.4
                    )
                )

                # Ajout des points d'entraînement
                fig.add_trace(
                    go.Scatter(
                        x=X_vis[:, 0],
                        y=X_vis[:, 1],
                        mode="markers",
                        marker=dict(color=y_vis, colorscale="Viridis", showscale=True),
                    )
                )

                # Configuration du graphique
                fig.update_layout(
                    title="Visualisation de l'Hypersplan SVM (Classification Multiclasse)",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    height=600,
                    width=600,
                )
                st.plotly_chart(fig)
        else:
            st.write(
                "Le modèle nécessite au moins 2 dimensions pour la visualisation de l'hyperplan."
            )
            # Export Functionality
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Export des Résultats
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour le rapport de classification</p>""",
            unsafe_allow_html=True,
        )
        report_filename = st.text_input("", value="rapport_classification.csv")

        st.markdown(
            """<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour le modèle</p>""",
            unsafe_allow_html=True,
        )
        model_filename = st.text_input("", value="model_svm.pkl")

        if st.button("Exporter les fichiers"):
            report_df.to_csv(report_filename, index=False)

            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {model_filename}"
            )

    if algo == "Naive Bayes":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Choix du modèle Naive Bayes
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:10px;">Sélectionnez le modèle Naive Bayes</p>""",
            unsafe_allow_html=True,
        )
        nb_model = st.sidebar.selectbox(
            "", options=["GaussianNB", "MultinomialNB", "BernoulliNB"]
        )

        # Appliquer MinMaxScaler pour garantir des valeurs positives

        if nb_model == "GaussianNB":
            model = GaussianNB()
        elif nb_model == "MultinomialNB":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = MultinomialNB()
        elif nb_model == "BernoulliNB":
            model = BernoulliNB()

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Évaluation des performances
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Accuracy :  <span style="color: #6ec3c2;">{acc:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Precision :  <span style="color: #6ec3c2;">{precision:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Recall :   <span style="color: #6ec3c2;">{recall:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">F1-score : <span style="color: #6ec3c2;">{f1:.2f}.</p>""",
            unsafe_allow_html=True,
        )

        # Rapport de classification en tableau
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.rename(columns={"index": "Classe"}, inplace=True)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Rapport de Classification
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            report_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Confusion
            </h3>""",
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        # Visualisation de la matrice de confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédictions", y="Véritables", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            height=600,  # Hauteur de l'image
            width=600,
        )
        fig.update_layout(title_text="", title_x=0.5)
        st.plotly_chart(fig)

        # Visualisation de la distribution des prédictions
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Distribution des Prédictions et des Véritables
            </h3>""",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=y_test, ax=ax, palette="coolwarm", label="Véritables")
        sns.countplot(x=y_pred, ax=ax, palette="Set1", alpha=0.5, label="Prédictions")
        ax.legend()
        st.pyplot(fig)
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
        model_filename = st.text_input("", value="model_naive_bayes.pkl")

        if st.button("Exporter les fichiers"):
            report_df = pd.DataFrame({"Véritables": y_test, "Prédictions": y_pred})
            importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": model.theta_[0]}
            )

            report_df.to_csv(report_filename, index=False)
            importance.to_csv(importance_filename, index=False)

            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    # Import nécessaire pour plot_tree
    from sklearn.tree import plot_tree

    if algo == "Decision Tree":
        # Séparation en données d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:10px;">Sélectionnez le modèle Decision Tree</p>""",
            unsafe_allow_html=True,
        )
        # Choix du type d'arbre de décision
        tree_type = st.sidebar.selectbox("", ["DecisionTreeClassifier", "CART"])

        if tree_type == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(random_state=42)
        elif tree_type == "CART":
            model = DecisionTreeClassifier(criterion="gini", random_state=42)

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Évaluation des performances
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Accuracy :  <span style="color: #6ec3c2;">{acc:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Precision :  <span style="color: #6ec3c2;">{precision:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Recall :   <span style="color: #6ec3c2;">{recall:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">F1-score : <span style="color: #6ec3c2;">{f1:.2f}.</p>""",
            unsafe_allow_html=True,
        )

        # Rapport de classification en tableau
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.rename(columns={"index": "Classe"}, inplace=True)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Rapport de Classification
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            report_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Confusion
            </h3>""",
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        # Visualisation de la matrice de confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédictions", y="Véritables", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            height=600,  # Hauteur de l'image
            width=600,
        )
        fig.update_layout(title_text="", title_x=0.5)
        st.plotly_chart(fig)

        # Importance des coefficients
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Importance des caractéristiques
            </h3>""",
            unsafe_allow_html=True,
        )
        feature_importance = pd.DataFrame(
            {"Caractéristique": X.columns, "Importance": model.feature_importances_}
        )
        feature_importance = feature_importance.sort_values(
            by="Importance", ascending=False
        )

        # Graphique de l'importance des caractéristiques
        fig_importance = px.bar(
            feature_importance,
            x="Importance",
            y="Caractéristique",
            orientation="h",
            title="",
        )
        st.plotly_chart(fig_importance)

        # Visualisation de l'arbre de décision
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Visualisation de l'Arbre de Décision
            </h3>""",
            unsafe_allow_html=True,
        )

        # Créer la figure
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            filled=True,
            feature_names=X.columns,
            class_names=[str(i) for i in np.unique(y)],
            rounded=True,
            ax=ax,
        )
        st.pyplot(fig)
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
        model_filename = st.text_input("", value="model_decision_tree.pkl")

        if st.button("Exporter les fichiers"):
            report_df.to_csv(report_filename, index=False)
            feature_importance = pd.DataFrame(
                {"Caractéristique": X.columns, "Importance": model.feature_importances_}
            )
            feature_importance.to_csv(importance_filename, index=False)

            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "Random Forest":
        # Diviser le dataset en train et test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Choix du modèle : Random Forest
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:10px;">Sélectionnez le nombre d'arbres</p>""",
            unsafe_allow_html=True,
        )
        n_estimators = st.sidebar.slider("", 10, 200, 100)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:10px;">Sélectionnez la profondeur maximale de l'arbre</p>""",
            unsafe_allow_html=True,
        )
        max_depth = st.sidebar.slider("", 1, 20, 10)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        rf_model.fit(X_train, y_train)

        # Prédictions
        y_pred = rf_model.predict(X_test)

        # Évaluation des performances
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Accuracy :  <span style="color: #6ec3c2;">{acc:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Precision :  <span style="color: #6ec3c2;">{precision:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Recall :   <span style="color: #6ec3c2;">{recall:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">F1-score : <span style="color: #6ec3c2;">{f1:.2f}.</p>""",
            unsafe_allow_html=True,
        )

        # Rapport de classification en tableau
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.rename(columns={"index": "Classe"}, inplace=True)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Rapport de Classification
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            report_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Confusion
            </h3>""",
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        # Visualisation de la matrice de confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédictions", y="Véritables", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            height=600,  # Hauteur de l'image
            width=600,
        )
        fig.update_layout(title_text="", title_x=0.5)
        st.plotly_chart(fig)

        # Importance des coefficients
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Importance des caractéristiques
            </h3>""",
            unsafe_allow_html=True,
        )

        # Visualisation de l'importance des caractéristiques
        feature_importance = pd.DataFrame(
            {"Caractéristique": X.columns, "Importance": rf_model.feature_importances_}
        )
        feature_importance = feature_importance.sort_values(
            by="Importance", ascending=False
        )

        fig_importance = px.bar(
            feature_importance,
            x="Importance",
            y="Caractéristique",
            orientation="h",
            title="",
        )
        st.plotly_chart(fig_importance)

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
            rf_model.estimators_[tree_index],
            feature_names=X.columns,
            class_names=[str(c) for c in rf_model.classes_],
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
        model_filename = st.text_input("", value="model_random_forest.pkl")

        if st.button("Exporter les fichiers"):
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().reset_index()
            report_df.rename(columns={"index": "Classe"}, inplace=True)
            report_df.to_csv(report_filename, index=False)
            importance = pd.DataFrame(
                {
                    "Caractéristique": X.columns,
                    "Importance": rf_model.feature_importances_,
                }
            )
            importance.to_csv(importance_filename, index=False)

            with open(model_filename, "wb") as f:
                pickle.dump(rf_model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    if algo == "KNN":
        # Division en train et test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Vérifier si les données ont exactement deux dimensions pour la visualisation
        if X.shape[1] == 2:
            visualize_decision_boundary = True
        else:
            visualize_decision_boundary = False

        # Configuration du modèle par l'utilisateur
        st.sidebar.write("### Configuration du Modèle KNN")
        n_neighbors = st.sidebar.slider("Nombre de voisins (k)", 1, 20, value=5)
        weights = st.sidebar.selectbox(
            "Méthode de pondération", ["uniform", "distance"]
        )
        metric = st.sidebar.selectbox(
            "Métrique de distance", ["minkowski", "euclidean", "manhattan"]
        )

        # Création du modèle KNN
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric
        )
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Évaluation des performances
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Performances du modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Accuracy :  <span style="color: #6ec3c2;">{acc:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Precision :  <span style="color: #6ec3c2;">{precision:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">Recall :   <span style="color: #6ec3c2;">{recall:.2f}.</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace;">F1-score : <span style="color: #6ec3c2;">{f1:.2f}.</p>""",
            unsafe_allow_html=True,
        )

        # Rapport de classification en tableau
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.rename(columns={"index": "Classe"}, inplace=True)
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Rapport de Classification
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            report_df.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de confusion
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Matrice de Confusion
            </h3>""",
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        # Visualisation de la matrice de confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédictions", y="Véritables", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            height=600,  # Hauteur de l'image
            width=600,
        )
        fig.update_layout(title_text="", title_x=0.5)
        st.plotly_chart(fig)

        # Visualisation de la frontière de décision (si applicable)
        if visualize_decision_boundary:
            st.markdown(
                """<h3 style="color: #A96255;">
                <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                Visualisation de la Frontière de Décision
                </h3>""",
                unsafe_allow_html=True,
            )
            # Création de la grille pour les points
            x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
            )

            # Prédiction sur la grille
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Tracer la frontière de décision
            fig_decision = px.imshow(
                Z,
                origin="lower",
                x=np.arange(x_min, x_max, 0.01),
                y=np.arange(y_min, y_max, 0.01),
                color_continuous_scale="Viridis",
                labels=dict(x=X.columns[0], y=X.columns[1], color="Classe prédite"),
            )
            fig_decision.update_traces(opacity=0.5)

            # Ajout des points de données
            scatter_points = px.scatter(
                x=X.iloc[:, 0],
                y=X.iloc[:, 1],
                color=pd.Categorical(y).codes,
                labels={"color": "Classe réelle"},
                title="Points et frontière de décision",
            )
            fig_decision.add_trace(scatter_points.data[0])
            st.plotly_chart(fig_decision)
        else:
            st.warning(
                "La visualisation de la frontière de décision est uniquement disponible pour des données avec 2 caractéristiques."
            )
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
        model_filename = st.text_input("", value="model_knn.pkl")

        if st.button("Exporter les fichiers"):
            # Exporter le rapport de classification
            report_df = pd.DataFrame(report).transpose().reset_index()
            report_df.rename(columns={"index": "Classe"}, inplace=True)
            report_df.to_csv(report_filename, index=False)

            # Exporter les coefficients
            importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": model.feature_importances_}
            )
            importance.to_csv(importance_filename, index=False)

            # Exporter le modèle
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {importance_filename}, {model_filename}"
            )

    st.markdown(
        """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Tester le Modèle
            </h3>""",
        unsafe_allow_html=True,
    )

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

        # Création d'un DataFrame test avec les colonnes sélectionnées
        # Création d'un DataFrame test avec les colonnes sélectionnées
        if selected_columns:
            # DataFrame global pour conserver les lignes précédentes
            if "df_test_global" not in st.session_state:
                st.session_state.df_test_global = pd.DataFrame(columns=selected_columns)

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
                    st.session_state.df_test_global = pd.concat(
                        [st.session_state.df_test_global, new_row],
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
            st.dataframe(st.session_state.df_test_global)

            # Prédiction sur les données test
            if st.button("Prédire"):
                df_test = st.session_state.df_test_global.copy()

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
