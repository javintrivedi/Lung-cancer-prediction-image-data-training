"""
export_models.py — Run this script INSIDE each Jupyter notebook to save
your trained models so the Flask backend can use them for real inference.

Copy and paste each cell into the appropriate notebook and run it.
"""

# ─────────────────────────────────────────────────
# CELL FOR: Cancer_Prediction_using_vgg16_Lung_Cancer_Image_dataset.ipynb
# Add this cell at the END of the VGG16 notebook, after model training.
# ─────────────────────────────────────────────────
VGG16_EXPORT_CELL = """
import os
os.makedirs("../backend/models", exist_ok=True)
model.save("../backend/models/vgg16_model.h5")
print("✅ VGG16 model saved to backend/models/vgg16_model.h5")
"""

# ─────────────────────────────────────────────────
# CELL FOR: Cancer_prediction_KNN_Lung_Cancer_image_dataset.ipynb
# Add this cell at the END of the KNN notebook, after model training.
# ─────────────────────────────────────────────────
KNN_EXPORT_CELL = """
import joblib, os
os.makedirs("../backend/models", exist_ok=True)

# Save the best KNN model (from GridSearchCV or your trained model variable)
# Replace 'best_knn' with your actual trained model variable name if different
joblib.dump(best_knn, "../backend/models/knn_model.pkl")
print("✅ KNN model saved to backend/models/knn_model.pkl")

# Also save the PCA/scaler if used for preprocessing
# joblib.dump(pca, "../backend/models/knn_pca.pkl")
# joblib.dump(scaler, "../backend/models/knn_scaler.pkl")
"""

# ─────────────────────────────────────────────────
# CELL FOR: Cancer_prediction_SVM.ipynb
# Add this cell at the END of the SVM notebook.
# ─────────────────────────────────────────────────
SVM_EXPORT_CELL = """
import joblib, os
os.makedirs("../backend/models", exist_ok=True)

# Replace 'svm_model' or 'clf' with your actual trained SVM variable name
joblib.dump(svm_model, "../backend/models/svm_model.pkl")
print("✅ SVM model saved to backend/models/svm_model.pkl")

# Save preprocessors too if used
# joblib.dump(pca, "../backend/models/svm_pca.pkl")
# joblib.dump(scaler, "../backend/models/svm_scaler.pkl")
"""

# ─────────────────────────────────────────────────
# CELL FOR: Cancer_prediction_hybrid.ipynb
# Add this cell at the END of the Hybrid CNN-SVM notebook.
# ─────────────────────────────────────────────────
HYBRID_EXPORT_CELL = """
import joblib, os
os.makedirs("../backend/models", exist_ok=True)

# Save the CNN feature extractor (Keras model)
cnn_feature_extractor.save("../backend/models/hybrid_cnn.h5")

# Save the SVM classifier on top
joblib.dump(hybrid_svm, "../backend/models/hybrid_svm.pkl")
print("✅ Hybrid CNN-SVM model saved to backend/models/")
"""

if __name__ == "__main__":
    print("This file is a reference guide.")
    print("Copy the export cells into each notebook and run them.")
    print("\nVGG16 notebook cell:")
    print(VGG16_EXPORT_CELL)
    print("\nKNN notebook cell:")
    print(KNN_EXPORT_CELL)
    print("\nSVM notebook cell:")
    print(SVM_EXPORT_CELL)
    print("\nHybrid notebook cell:")
    print(HYBRID_EXPORT_CELL)
