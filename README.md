# Matrix Factorization Recommender System (NumPy Implementation)

A from-scratch implementation of a collaborative filtering recommender system using matrix factorization and gradient descent, trained on the MovieLens 100k dataset.

## 📋 Project Overview

This project demonstrates how modern recommender systems work by implementing **matrix factorization** from scratch using only NumPy. Instead of relying on high-level libraries, this implementation provides a deep understanding of the mathematical foundations behind recommendation algorithms used by companies like Netflix, Amazon, and Spotify.

The model learns latent representations of users and movies through collaborative filtering, predicting missing ratings by factorizing the user-item rating matrix into two lower-dimensional matrices.

## ✨ Key Features

- **Pure NumPy Implementation**: Complete matrix factorization built from scratch without using high-level ML frameworks
- **Gradient Descent Optimization**: Custom implementation of stochastic gradient descent for model training
- **MSE Loss Calculation**: Mean squared error metric for evaluating prediction accuracy
- **MovieLens 100k Dataset**: Trained on a real-world dataset containing 100,000 movie ratings
- **Visualization**: Training loss curves and error distribution analysis
- **Mathematical Derivation**: Includes symbolic gradient computation using SymPy for educational purposes

## 🎯 Model Architecture

The model factorizes the rating matrix $R$ into two matrices:
- $P$ (User Features): $n_{users} \times k$ matrix representing user preferences
- $Q$ (Movie Features): $n_{movies} \times k$ matrix representing movie characteristics

The prediction is computed as: $\hat{R} = P Q^T$

We minimize the squared error objective:
$$\min_{P, Q} \sum_{(u, i) \in \Omega} (R_{ui} - P_u^T Q_i)^2$$

where $\Omega$ is the set of observed ratings.

## 📊 Results

- **Test Set MSE**: ~0.91
- **Training Epochs**: 100
- **Learning Rate**: 0.001
- **Latent Factors**: 5

The model successfully learns to predict user ratings with reasonable accuracy, demonstrating the effectiveness of matrix factorization for collaborative filtering.

## 🚀 Installation & Usage

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/numpy-matrix-factorization-recommender.git
cd numpy-matrix-factorization-recommender
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Matrix_Factorization_Recommender.ipynb
```

### Running the Code

The notebook will automatically:
1. Download the MovieLens 100k dataset
2. Split data into train/test sets
3. Initialize the matrix factorization model
4. Train using gradient descent
5. Evaluate performance on the test set
6. Visualize training progress and error distributions

## 📁 Project Structure

```
.
├── Matrix_Factorization_Recommender.ipynb   # Main implementation notebook
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
├── .gitignore                               # Git ignore patterns
└── ml-100k/                                 # MovieLens dataset (auto-downloaded)
```

## 🧠 Learning Outcomes

By exploring this project, you will understand:
- How collaborative filtering works mathematically
- The principles behind matrix factorization
- Implementation of gradient descent from scratch
- Latent factor models in recommender systems
- Evaluation metrics for recommendation quality

## 🔬 Technical Details

**Algorithm**: Stochastic Gradient Descent  
**Loss Function**: Mean Squared Error (MSE)  
**Dataset**: MovieLens 100k (943 users, 1,682 movies, 100,000 ratings)  
**Train/Test Split**: 80/20 (stratified by user)

## 📝 License

This project is available under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- GroupLens Research for their work in recommender systems

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [your email].

## 🔗 References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*, 263-272.

---

*Built with ❤️ using NumPy and Python*
