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
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import pickle
from PIL import Image
import os


def NeuralNetworks(df, target):
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
        "<h3 class='miniTitle' style ='margin-left:455px; margin-top:-30px;color: #00A8A8;'>Deep Learning</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:30px;">Sélectionnez l'algorithme souhaité</p>""",
        unsafe_allow_html=True,
    )
    algo = st.sidebar.selectbox("", options=["Default", "ANN", "CNN"])

    df = df.dropna()
    X = df.drop(columns=[target])  # Exclure la colonne cible
    y = df[target]  # La colonne cible

    # Traitement des colonnes numériques (normalisation)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Traitement des colonnes non numériques (encodage)
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    # Encodage de la cible (si elle est catégorielle)
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    if algo == "ANN":
        # Séparation en données d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Création du modèle ANN
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:30px;">Nombre de couches cachées</p>""",
            unsafe_allow_html=True,
        )
        layers = st.sidebar.slider("", 1, 5, 2)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px;">Nombre de neurones par couche</p>""",
            unsafe_allow_html=True,
        )
        neurons = st.sidebar.slider("", 10, 100, 64)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px;">Fonction d'activation</p>""",
            unsafe_allow_html=True,
        )
        activation = st.sidebar.selectbox("", ["relu", "tanh", "sigmoid"])
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px;">Optimiseur</p>""",
            unsafe_allow_html=True,
        )
        optimizer = st.sidebar.selectbox("", ["adam", "sgd", "rmsprop"])
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px;">Nombre d'époques</p>""",
            unsafe_allow_html=True,
        )
        epochs = st.sidebar.slider("", 10, 100, 50)
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px;">Taille du batch</p>""",
            unsafe_allow_html=True,
        )
        batch_size = st.sidebar.slider("", 16, 128, 32)

        # Construction du modèle
        model = Sequential()
        model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))

        for _ in range(layers - 1):
            model.add(Dense(neurons, activation=activation))

        # Activation de la dernière couche en fonction de la classification binaire ou multi-classes
        output_activation = "sigmoid" if len(y.unique()) == 2 else "softmax"
        model.add(
            Dense(
                1 if len(y.unique()) == 2 else len(y.unique()),
                activation=output_activation,
            )
        )

        # Compilation du modèle
        loss_function = (
            "binary_crossentropy"
            if len(y.unique()) == 2
            else "categorical_crossentropy"
        )
        model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        # Prédictions
        y_pred = model.predict(X_test)
        if len(y.unique()) == 2:
            y_pred = (y_pred > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred, axis=1)

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

        # Affichage des courbes d'apprentissage
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Courbes d'apprentissage
            </h3>""",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(history.history["accuracy"], label="Précision entraînement")
        ax[0].plot(history.history["val_accuracy"], label="Précision validation")
        ax[0].set_title("Précision")
        ax[0].legend()

        ax[1].plot(history.history["loss"], label="Perte entraînement")
        ax[1].plot(history.history["val_loss"], label="Perte validation")
        ax[1].set_title("Perte")
        ax[1].legend()

        st.pyplot(fig)
        # --- Section Export des résultats ---
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Export des Résultats
            </h3>""",
            unsafe_allow_html=True,
        )

        # Input pour les noms des fichiers
        st.markdown(
            """<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour le rapport de classification</p>""",
            unsafe_allow_html=True,
        )
        report_filename = st.text_input(
            "",
            value="rapport_classification.csv",
        )
        st.markdown(
            """<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour les coefficients d'importance</p>""",
            unsafe_allow_html=True,
        )
        importance_filename = st.text_input("", value="coefficients_importance.csv")
        st.markdown(
            """<p style="font-size: 20px; color: #f7dace;margin-bottom:-30px;">Nom du fichier pour le modèle</p>""",
            unsafe_allow_html=True,
        )
        model_filename = st.text_input("", value="model_ann.pkl")

        # Bouton d'exportation
        if st.button("Exporter les fichiers"):
            # Exporter le rapport de classification
            report_df.to_csv(report_filename, index=False)

            # Exporter les coefficients (optionnel : ajuster en fonction de vos données)
            importance_df = pd.DataFrame(model.get_weights())  # Exemple
            importance_df.to_csv(importance_filename, index=False)

            # Exporter le modèle entraîné
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : "
                f"{report_filename}, {importance_filename}, {model_filename}"
            )
        # Titre de l'interface
        st.markdown(
            """<h3 style="color: #A96255;">
                <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                Tester le Modèle
            </h3>""",
            unsafe_allow_html=True,
        )

        # Initialisation de df_test_global_ANN
        df_test_global_ANN = pd.DataFrame()

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
                if df_test_global_ANN.empty:
                    df_test_global_ANN = pd.DataFrame(columns=selected_columns)

                # Initialisation du dictionnaire temporaire avec les colonnes sélectionnées
                temp_row = {column: "" for column in selected_columns}

                # Formulaire d'insertion des valeurs
                st.markdown(
                    f"""<p style="font-size: 20px; color: #f7dace;">Entrez les valeurs pour chaque colonne :</p>""",
                    unsafe_allow_html=True,
                )

                # Collecte des valeurs des colonnes sélectionnées
                for column in selected_columns:
                    temp_row[column] = st.text_input(
                        f"Entrez une valeur pour {column} :",
                        temp_row[column],
                        key=f"test_input_{column}",
                    )

                # Bouton pour valider l'ajout de la ligne
                if st.button("Ajouter la ligne"):
                    # Vérification que toutes les colonnes ont une valeur
                    if all(value != "" for value in temp_row.values()):
                        # Ajout de la ligne au DataFrame global
                        new_row = pd.DataFrame([temp_row])
                        df_test_global_ANN = pd.concat(
                            [df_test_global_ANN, new_row],
                            ignore_index=True,
                        )
                        # Réinitialisation du dictionnaire temporaire
                        temp_row = {column: "" for column in selected_columns}
                        st.success("Ligne ajoutée avec succès !")
                    else:
                        st.error(
                            "Veuillez remplir toutes les colonnes avant d'ajouter une ligne."
                        )

                st.markdown(
                    f"""<p style="font-size: 20px; color: #f7dace;">Aperçu des données de test :</p>""",
                    unsafe_allow_html=True,
                )
                st.dataframe(df_test_global_ANN)

                # Vérifier si des données sont présentes avant la prédiction
                if df_test_global_ANN.empty:
                    st.error("Aucune donnée test n'est disponible.")
                else:
                    # Prédiction sur les données test
                    if st.button("Prédire"):
                        df_test = df_test_global_ANN.copy()

                        if not df_test.empty:
                            # Encodage des colonnes catégoriques dans le jeu de test
                            for col in categorical_cols:
                                if col in df_test.columns:
                                    le = LabelEncoder()
                                    le.fit(
                                        df[col].astype(str)
                                    )  # Fit sur les données d'entraînement
                                    df_test[col] = df_test[col].map(
                                        lambda x: (
                                            le.transform([x])[0]
                                            if x in le.classes_
                                            else -1
                                        )
                                    )

                            # Gestion des colonnes manquantes dans df_test
                            for col in X.columns:
                                if col not in df_test.columns:
                                    df_test[col] = (
                                        0  # Remplir avec une valeur par défaut
                                    )

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

    if algo == "CNN":
        # Séparer en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Convertir X_train et X_test en tableaux NumPy avant de les normaliser
        X_train = X_train.values
        X_test = X_test.values

        # Normalisation des données
        X_train, X_test = X_train / 255.0, X_test / 255.0

        # Vérifier la forme des données
        print("Shape of X_train before reshape:", X_train.shape)
        print("Shape of X_test before reshape:", X_test.shape)

        # Redimensionner pour les données d'entrée de convolution (28x28x1)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        # Vérifier la forme après reshape
        print("Shape of X_train after reshape:", X_train.shape)
        print("Shape of X_test after reshape:", X_test.shape)

        # Définir le nombre de classes
        num_classes = len(np.unique(y))  # Nombre de classes uniques dans y

        # Créer le modèle CNN
        model = Sequential()
        model.add(
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding="same"
            )
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
        )

        # Prédictions et évaluation
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        # Évaluation des performances
        test_loss, test_acc = model.evaluate(X_test, y_test)
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
            f"""<p style="font-size: 20px; color: #f7dace;">Loss :  <span style="color: #6ec3c2;">{test_loss:.4f}.</p>""",
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

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y),
        )
        plt.xlabel("Prédictions")
        plt.ylabel("Véritables")
        st.pyplot(fig)

        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Courbe d'entraînement
            </h3>""",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots()
        ax.plot(history.history["accuracy"], label="Accuracy")
        ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)
        # Section d'exportation des fichiers
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Export des Résultats
            </h3>""",
            unsafe_allow_html=True,
        )

        # Fichier pour le rapport de classification
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour le rapport de classification</p>""",
            unsafe_allow_html=True,
        )
        report_filename = st.text_input("", value="rapport_classification.csv")

        # Fichier pour les coefficients
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour les coefficients</p>""",
            unsafe_allow_html=True,
        )
        importance_filename = st.text_input("", value="coefficients_importance.csv")

        # Fichier pour le modèle
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour le modèle</p>""",
            unsafe_allow_html=True,
        )
        model_filename = st.text_input("", value="model_cnn.pkl")

        # Fichier pour les visualisations (courbe d'entraînement)
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">Nom du fichier pour la courbe d'entraînement</p>""",
            unsafe_allow_html=True,
        )
        curve_filename = st.text_input("", value="training_curve.png")

        if st.button("Exporter les fichiers"):
            # Exporter le rapport de classification
            report_df.to_csv(report_filename, index=False)

            # Exporter les coefficients (si disponibles)
            # importance.to_csv(importance_filename, index=False)  # Exemple si vous avez un dataframe 'importance'

            # Exporter le modèle
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            # Exporter la courbe d'entraînement
            fig.savefig(curve_filename)

            st.success(
                f"Les fichiers ont été exportés avec succès sous les noms : {report_filename}, {model_filename}, {curve_filename}"
            )
        st.markdown(
            """<h3 style="color: #A96255;">
                <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                Tester le Modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            """<h3 style="color: #A96255;">
                <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px; margin-bottom: -30px;"></i>
                Tester le Modèle
            </h3>""",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px; margin-top:30px;">Sélectionnez la pièce souhaitée</p>""",
            unsafe_allow_html=True,
        )
        choix = st.sidebar.selectbox(
            "",
            options=[
                "Default",
                "T-shirt",
                "Pantalon",
                "Pull",
                "Robe",
                "Manteau",
                "Sandale",
                "Chemise",
                "Basket",
                "Sac",
                "Boutine",
            ],
        )
        file_path = False
        if choix:
            if choix == "Boutine":
                file_path = "assets/boutine.jpg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "T-shirt":
                file_path = "assets/T-shirt.jpg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Pantalon":
                file_path = "assets/pantalon.jpeg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Pull":
                file_path = "assets/Pull.jpg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Robe":
                file_path = "assets/dress.jpg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Manteau":
                file_path = "assets/manteau.jpeg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Sandale":
                file_path = "assets/sandale.jpeg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Chemise":
                file_path = "assets/chemise.jpeg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Basket":
                file_path = "assets/bascket.jpeg"
                st.sidebar.image(file_path, use_container_width=True, caption="")
            elif choix == "Sac":
                file_path = "assets/sac.jpeg"
                st.sidebar.image(file_path, use_container_width=True, caption="")

            if file_path:
                # Chargement l'image depuis le chemin specifie
                im = Image.open(file_path)
                # Convertissement l'image en niveaux de gris
                im_gris = im.convert("L")
                # Redimensionnement l'image a la taille en 28x28 pixels
                im_red = im_gris.resize((28, 28))
                # Obtenir les valeurs des pixels sous forme de tableau 1D
                pixel_array = list(im_red.getdata())
                # Assurer que le tableau a la taille attendue (784 pixels)
                if len(pixel_array) != 28 * 28:
                    raise ValueError("L'image n'a pas la taille attendue.")
                # Convertir la liste en un tableau numpy et appliquer la transformation sur les donnees
                pixel_array_scaled = np.array(pixel_array) / 255.0
                # Redimensionner le tableau pour qu'il ait la forme appropriée pour l'entree du modele CNN
                pixel_array_scaled = pixel_array_scaled.reshape(1, 28, 28, 1)
                # Faire une prediction avec le modele CNN
                Y_prevu = model.predict(pixel_array_scaled)
                # Prendre la plus grande valeur de probabilite pour savoir la classe exacte
                max = Y_prevu[0][0]
                x = 0
                for i in range(len(Y_prevu[0])):
                    if max < Y_prevu[0][i]:
                        max = Y_prevu[0][i]
                        x = i
                # Le texte affiche
                if x == 0:
                    text = "Image inserée est un T-shirt"
                elif x == 1:
                    text = "Image inserée est un Pantalon"
                elif x == 2:
                    text = "Image inserée est un Pull"
                elif x == 3:
                    text = "Image inserée est une Robe"
                elif x == 4:
                    text = "Image inserée est un Manteau"
                elif x == 5:
                    text = "Image inserée est une Sandale"
                elif x == 6:
                    text = "Image inserée est une Chemise"
                elif x == 7:
                    text = "Image inserée est une Basket"
                elif x == 8:
                    text = "Image inserée est un Sac"
                else:
                    text = "Image inserée est une Bottine"
                # Retournant le text predit
                st.markdown(
                    f"""<p style="font-size: 20px; color: #f7dace; margin-bottom:-30px;">{text}</p>""",
                    unsafe_allow_html=True,
                )
