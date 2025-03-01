import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gerando um dataset fictício para e-commerce
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "preco_produto": np.random.uniform(50, 500, n_samples),
    "estoque_disponivel": np.random.randint(10, 500, n_samples),
    "trafego_site": np.random.randint(100, 10000, n_samples),
    "avaliacoes_positivas": np.random.randint(1, 500, n_samples),
    "desconto_aplicado": np.random.uniform(0, 0.5, n_samples),  # Percentual de desconto
})

# Criando a variável de vendas com base nas outras variáveis + um fator aleatório
data["vendas"] = (data["trafego_site"] * 0.005 +
                  data["avaliacoes_positivas"] * 3 +
                  data["preco_produto"] * -0.2 +
                  data["desconto_aplicado"] * 200 +
                  np.random.normal(0, 50, n_samples))

# Separando variáveis independentes (X) e dependente (y)
X = data.drop(columns=["vendas"])
y = data["vendas"]

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão linear múltipla
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Avaliando o modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Exibindo os resultados
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualizando a relação entre valores reais e preditos
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Vendas Reais")
plt.ylabel("Vendas Previstas")
plt.title("Comparação entre valores reais e previstos")
plt.show()

# Salvando os dados gerados para upload no GitHub
data.to_csv("ecommerce_sales_data.csv", index=False)



git config --global user.email "mialiberata@gmail.com"
git config --global user.name "mialiberata"