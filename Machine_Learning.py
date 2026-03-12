import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Abre o arquivo onde estão os dados históricos dos clientes
tabela = pd.read_csv('clientes.csv')

# 2. Remove qualquer linha que tenha informações faltando (NaN)
# Isso evita que o modelo "se engasgue" com dados incompletos
tabela = tabela.dropna()

# 3. O "Cérebro Digital" só entende números, não entende palavras.
# Esse bloco percorre as colunas de texto (ex: Profissão) e troca cada palavra por um número
codificador = LabelEncoder()
for coluna in tabela.columns:
    # Se a coluna não for um número (for texto), a gente transforma em número
    if not pd.api.types.is_numeric_dtype(tabela[coluna]):
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

# 4. Define quem é quem:
# X = As "pistas" (idade, salário, etc). Tiramos o 'id_cliente' porque é só um número que não ajuda a prever nada.
# Y = A "resposta" que queremos que ele aprenda a dar (o Score de Crédito)
x = tabela.drop(['score_credito', 'id_cliente'], axis=1)
y = tabela['score_credito']

# 5. Separa a base em duas partes:
# Uma parte (70%) para o modelo estudar (treino) e outra (30%) para fazermos uma prova depois (teste)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

# 6. Cria e treina a inteligência artificial (Floresta Aleatória)
# É aqui que ele começa a procurar padrões entre os dados e o score
modelo = RandomForestClassifier()
modelo.fit(x_treino, y_treino)

# 7. Hora da prova: damos os dados de teste (sem as respostas) para o modelo
# Ele diz o que acha que é, e depois comparamos com a resposta real para ver a porcentagem de acerto
previsoes = modelo.predict(x_teste)
print(f"Acurácia do modelo: {accuracy_score(y_teste, previsoes):.2%}")
