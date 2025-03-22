# Rede Neural - Idéias
### geral
- Normalizar as notas (0 a 1), (0 a 4) ou (0 a 5) - a decidir
- Calcular média final dando peso as notas atribuidas por cada uma das correções
- Entrada para a rede neural: as 3 notas e a média (features para rede)
- Variável target: nota do professor
- Separar os dados entre treino, validação e teste

### Construção da rede neural
- Entrada: 3 notas e a média
- Camadas: Algumas camadas com ativação ReLU para aprender os padrões
- Saída: Neurônio único com ativação linear para prever a nota final
- Perda: erro quadrático médio (MSE) para minimizar a diferença entre o valor previsto e a nota real

### Treinamento e avaliação
- Usar um otimizador Adam no treinamento
- Utilizar early stopping para evitar overfitting
- Avalie com métricas como RMSE ou MAE para medir a proximidade das previsões

