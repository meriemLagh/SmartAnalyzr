import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from scipy import stats
from sklearn.decomposition import PCA
import os


def NettoyageDataset(df, target):
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

    st.markdown(
        "<h3 class='miniTitle' style ='margin-left:410px; margin-top:-40px;color: #00A8A8;'>Nettoyage des données!</h3>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<h2 style="color: #e97f4f; margin-bottom: -40px; text-align : center; font-size: 20px;">Les options disponibles</h2>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<p style="font-size: 15px; color: #f7dace; margin-top : 20px; margin-bottom: -35px;">Sélectionnez le type de modèle appliqué</p>""",
        unsafe_allow_html=True,
    )
    type = st.sidebar.selectbox("", options=["Machine Learning", "Deep Learning"])

    st.markdown(
        """<h3 style="color: #A96255; margin-top:30px;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Affichage
        </h3>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<p style="font-size: 20px; color: #f7dace; margin-bottom: -40px;">Affichage des 10 premiers lignes du dataset avant le nettoyage</p>""",
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
    df_copy = df.copy()
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.markdown(
        df_copy.head(10).to_html(classes="dataframe", index=False),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    checkNaN = st.sidebar.checkbox("Gestion des valeurs manqunates")
    if checkNaN:
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # Remplacer les valeurs nulles des colonnes catégorielles par la valeur la plus fréquente
        for col in df.select_dtypes(include=["object"]).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

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
        st.markdown(
            som.to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    checkDup = st.sidebar.checkbox("Gestion des valeurs doublantes")
    if checkDup:
        df.drop_duplicates(inplace=True)
        # Vérifier les doublons
        st.markdown(
            """<h3 style="color: #A96255;">
            <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
            Vérification des doublons
            </h3>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<p style="font-size: 20px; color: #f7dace; ">Nombre de doublons dans le dataset: {df.duplicated().sum()}</p>""",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        """<p style="font-size: 15px; color: #f7dace; margin-top : 20px; margin-bottom: -35px;">Sélectionnez la méthode de normalisation souhaitée</p>""",
        unsafe_allow_html=True,
    )
    norm = st.sidebar.selectbox(
        "",
        options=[
            "Default",
            "MinMaxScaler",
            "StandardScaler",
            "RobustScaler",
            "MaxAbsScaler",
        ],
    )

    copy = df.copy()
    if target is not None:
        X = df.drop(columns=[target])  # Exclure la colonne cible
    else:
        X = df.copy()
    numeric_cols = X.select_dtypes(
        include=[np.number]
    ).columns  # Garder uniquement les colonnes numériques
    # Appliquer la normalisation uniquement sur les colonnes numériques
    if norm == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(
            X[numeric_cols]
        )  # Normaliser seulement les colonnes numériques
        df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
        if target is not None:
            df_scaled[target] = df[target]  # Réintégrer la colonne cible
        df = df_scaled.copy()
        if target is not None:
            if len(df[target].unique() < 10):  # Si toutes les valeurs sont des entiers
                if df[target].dtype in ["float64", "int64"]:
                    df[target].fillna(df[target].mean(), inplace=True)

                # Remplacer les valeurs nulles des colonnes catégorielles par la valeur la plus fréquente
                if df[target].dtype in ["object"]:
                    df[target].fillna(df[target].mode()[0], inplace=True)
                df[target] = df[target].astype(int)  # Convertir la colonne en entier
        missing_columns = [col for col in copy.columns if col not in df.columns]

        # Ajouter les colonnes manquantes à df
        df = df.join(copy[missing_columns], how="left")
        # Affichage du DataFrame après normalisation
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            df.head(5).to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    elif norm == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(
            X[numeric_cols]
        )  # Normaliser seulement les colonnes numériques
        df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
        if target is not None:
            df_scaled[target] = df[target]  # Réintégrer la colonne cible
        df = df_scaled.copy()
        if target is not None:
            if len(df[target].unique() < 10):  # Si toutes les valeurs sont des entiers
                if df[target].dtype in ["float64", "int64"]:
                    df[target].fillna(df[target].mean(), inplace=True)

                # Remplacer les valeurs nulles des colonnes catégorielles par la valeur la plus fréquente
                if df[target].dtype in ["object"]:
                    df[target].fillna(df[target].mode()[0], inplace=True)
                df[target] = df[target].astype(int)  # Convertir la colonne en entier
        missing_columns = [col for col in copy.columns if col not in df.columns]

        # Ajouter les colonnes manquantes à df
        df = df.join(copy[missing_columns], how="left")
        # Affichage du DataFrame après normalisation
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            df.head(5).to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    elif norm == "RobustScaler":
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(
            X[numeric_cols]
        )  # Normaliser seulement les colonnes numériques
        df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
        if target is not None:
            df_scaled[target] = df[target]  # Réintégrer la colonne cible
        df = df_scaled.copy()
        if target is not None:
            if len(df[target].unique() < 10):  # Si toutes les valeurs sont des entiers
                if df[target].dtype in ["float64", "int64"]:
                    df[target].fillna(df[target].mean(), inplace=True)

                # Remplacer les valeurs nulles des colonnes catégorielles par la valeur la plus fréquente
                if df[target].dtype in ["object"]:
                    df[target].fillna(df[target].mode()[0], inplace=True)
                df[target] = df[target].astype(int)  # Convertir la colonne en entier
        missing_columns = [col for col in copy.columns if col not in df.columns]

        # Ajouter les colonnes manquantes à df
        df = df.join(copy[missing_columns], how="left")
        # Affichage du DataFrame après normalisation
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            df.head(5).to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    elif norm == "MaxAbsScaler":
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(
            X[numeric_cols]
        )  # Normaliser seulement les colonnes numériques
        df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
        if target is not None:
            df_scaled[target] = df[target]  # Réintégrer la colonne cible
        df = df_scaled.copy()
        if target is not None:
            if len(df[target].unique() < 10):  # Si toutes les valeurs sont des entiers
                if df[target].dtype in ["float64", "int64"]:
                    df[target].fillna(df[target].mean(), inplace=True)

                # Remplacer les valeurs nulles des colonnes catégorielles par la valeur la plus fréquente
                if df[target].dtype in ["object"]:
                    df[target].fillna(df[target].mode()[0], inplace=True)
                df[target] = df[target].astype(int)  # Convertir la colonne en entier
        missing_columns = [col for col in copy.columns if col not in df.columns]

        # Ajouter les colonnes manquantes à df
        df = df.join(copy[missing_columns], how="left")
        # Affichage du DataFrame après normalisation
        st.markdown('<div class="center-table">', unsafe_allow_html=True)
        st.markdown(
            df.head(5).to_html(classes="dataframe", index=False), unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        df_scaled = (
            df.copy()
        )  # Si la normalisation n'est pas sélectionnée, retourner le DataFrame original

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df1 = (
        df.copy()
    )  # Copie initiale du DataFrame pour pouvoir réinitialiser après chaque encodage

    if len(categorical_cols) > 0:
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : 20px; margin-bottom: -35px;">Sélectionnez la méthode d'encodage souhaitée</p>""",
            unsafe_allow_html=True,
        )
        encoding = st.sidebar.selectbox(
            "", options=["Default", "LabelEncoder", "OneHotEncoder"]
        )

        # Réinitialiser le DataFrame à sa copie initiale avant chaque méthode d'encodage
        df = df1.copy()

        if encoding == "LabelEncoder":
            for col in categorical_cols:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                st.markdown(
                    """<h3 style="color: #A96255;">
                <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                Encodage
                </h3>""",
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="center-table">', unsafe_allow_html=True)
                st.markdown(
                    df.head().to_html(classes="dataframe", index=False),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        elif encoding == "OneHotEncoder":
            one_hot_encoded = pd.get_dummies(
                df[categorical_cols], prefix=categorical_cols
            )
            one_hot_encoded = one_hot_encoded.astype(int)
            df = pd.concat([df, one_hot_encoded], axis=1).drop(columns=categorical_cols)
            st.markdown(
                """<h3 style="color: #A96255;">
                <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                Encodage
                </h3>""",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="center-table">', unsafe_allow_html=True)
            st.markdown(
                df.head().to_html(classes="dataframe", index=False),
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        elif encoding == "Default":
            df = df1.copy()

    checkOut = st.sidebar.checkbox("Gestion des outliers")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if checkOut:
        st.sidebar.markdown(
            """<p style="font-size: 15px; color: #f7dace; margin-top : -20px; margin-bottom: -35px;">Sélectionnez la méthode de gestion d'outliers</p>""",
            unsafe_allow_html=True,
        )
        out = st.sidebar.selectbox("", options=["Default", "IQR", "Z-score"])
        if out == "IQR":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            st.markdown(
                """<h3 style="color: #A96255;">
                    <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                    Gestion d'outliers
                </h3>""",
                unsafe_allow_html=True,
            )
            st.success(
                "La gestion d'outliers avec la méthode Ecart Interquartile a été bien effectuée"
            )
        elif out == "Z-score":
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < 3]
            st.markdown(
                """<h3 style="color: #A96255;">
                    <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                    Gestion d'outliers
                </h3>""",
                unsafe_allow_html=True,
            )
            st.success(
                "La gestion d'outliers avec la méthode Z-score a été bien effectuée"
            )
    if type == "Machine Learning":
        if target is not None and df.shape[1] > 2:
            checkCorr = st.sidebar.checkbox(
                "Suppression des variables non significatives"
            )
            if checkCorr:
                # Définir un seuil de corrélation
                correlation_threshold = (
                    0.1  # Vous pouvez ajuster ce seuil selon vos besoins
                )
                # Calculer la corrélation entre toutes les colonnes et la variable cible
                correlations = df.corr()
                # Extraire la corrélation de la variable cible avec les autres colonnes
                target_correlation = correlations[target]
                # Sélectionner les colonnes dont la corrélation est supérieure au seuil
                cols_to_keep = target_correlation[
                    target_correlation.abs() >= correlation_threshold
                ].index
                # Garder seulement les colonnes ayant une corrélation suffisante avec la variable cible
                df = df[cols_to_keep]
                st.markdown(
                    """<h3 style="color: #A96255;">
                        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                        Etude de corrélation
                    </h3>""",
                    unsafe_allow_html=True,
                )
                df_copy = df.copy()
                st.markdown('<div class="center-table">', unsafe_allow_html=True)
                st.markdown(
                    df.head(5).to_html(classes="dataframe", index=False),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                dfB = df.copy()
        if df.shape[1] > 2:
            checkPCA = st.sidebar.checkbox("Application du méthode ACP")
            if checkPCA:
                X = df.drop(columns=[target])
                pca = PCA()
                pca.fit(X)
                # Calcul de la variance expliquée par chaque composante principale
                explained_variance = pca.explained_variance_ratio_

                # Calcul de la variance expliquée cumulée
                cumulative_variance = np.cumsum(explained_variance)
                n_components = np.argmax(cumulative_variance >= 0.95) + 1
                # Appliquer PCA avec le nombre optimal de composantes
                pca_optimal = PCA(n_components=n_components)
                df_pca = pca_optimal.fit_transform(X)
                # Assurez-vous que df_pca est un DataFrame
                df_pca = pd.DataFrame(df_pca)

                # Renommer les colonnes du DataFrame avec des noms comme "PC1", "PC2", etc.
                df_pca.columns = [f"PC{i+1}" for i in range(df_pca.shape[1])]

                # Ajouter la colonne cible (non transformée) à df_pca
                df_pca[target] = dfB[target].values

                st.markdown(
                    """<h3 style="color: #A96255;;">
                        <i class="fas fa-arrow-right" style="font-size: 20px; color: #e97f4f; margin-right: 10px;"></i>
                            PCA
                        </h3>""",
                    unsafe_allow_html=True,
                )
                df = df_pca.copy()
                st.markdown('<div class="center-table">', unsafe_allow_html=True)
                st.markdown(
                    df.head(5).to_html(classes="dataframe", index=False),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    return type
