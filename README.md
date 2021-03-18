# IC-2020-final

## Autor: José Ricardo Silveira

### Um compilado com os principais códigos que usei no final da IC 2020 no Grupo de Astrofísica da Universidade Federal de Santa Catarina.

*Notas* Não foi usado o git durante a execução do trabalho, tampouco foi formatado em PEP8, por desconhecimento. Recentemente adicionei modificações, majoritariamente, no espaçamento e docstrings para exercício. Entretanto, os nomes das variáveis permanecem os mesmos 👎.
 Acho importante destacar os pontos de melhoria encontrados para implementá-los em outros projetos.
 
A descrição de algumas funções e os nomes das variáveis estão em inglês para exercício, em trabalhos futuros, que necessitem de tal abordagem.

O objetivo era separar a enorme tabela, com mais de 3GB, de objetos do GAIA + S-PLUS + SDSS em tabelas menores, de modo que o treinamento pudesse ser feito em meu computador. Além disso, o tamanho de cada tabela foi muito menor. Por questões de espaço, essas tabelas estão armazenadas nos servidores locais da astro. Adicionalmente, todo o procedimento de merge dos banco de dados do GAIA, SDSS e S-PLUS estão descritos no relatório final. Com isso em mente, parto do pressuposto de operar essas tabelas menores, uma a uma, mas com dados equivalentes.

#### O trabalho está dividido em etapas. Primeiro, há a divisão das tabelas em diferentes features e testes.
  
  Um ponto a ser melhorado, era usar as variáveis categóricas de classificação dada aos objetos e transformá-las em dummies. Com isso, pode-se avaliar o impacto de determinadas classes de objetos, e.g, quasars, impactariam no reconhecimento.
  
  Também deve-se fixar uma semente geral para consistência dos resultados. Ela já está fixada, mas é declarada uma seed toda vez que for preciso. Basta usar a seed do numpy que fixa no algoritmo geral: np.random.seed(SEED) pois os algoritmos do SK-learn utilizam o numpy por baixo dos panos.

#### O coração do trabalho está na segunda etapa executada no arquivo train_test_predict_v4.py .

Aqui, é feito vários sorteios e tomado o resultado como conjunto oficial aquele que apresente valor mais próximo da média de acertos. São plotados ou salvos histogramas de média, mediana e desvio padrão de cada conjunto de treinamento e para cada conjunto de resultados: False positives, False negatives, True positives, True negatives. Para cada conjunto de features, são feitos 10, 100 e 1000 sorteios no treinamento. Por fim, são salvos DataFrames com os conjuntos que julgamos serem mais significativos GAIA, S-PLUS r, GAIA + S-PLUS r e cada coluna será uma categoria de resultado: False positives, False negatives, True positives, True negative.
  
  Pode-se salvar cada algoritmo treinado para prever outors conjuntos. Cada modelo de treinamento pode ser salvo para prever outros conjuntos. Isso pode ser feito usando o pickle. Mais informações aqui https://scikit-learn.org/stable/modules/model_persistence.html
  
#### Na terceira etapa, executa-se o arquivo sdss_spectra_v4.py, o qual baixa todos os espectros disponíveis dos conjuntos de treinamento como também registra os que não puderam ser baixados.

Além disso, habilitando as linhas de código que contenham a função sdss_spec, plota-se, para cada categoria de resultado de cada conjunto, os espectros disponíveis com a formatação apropriada. 

#### Na quarta e última etapa, há o plot_color_diagrams_v4.py, salvo três diagramas hertzsprung com diferentes medidas em cada magnitude. Em cada um, é destacado e identificado cada falso negativo.

#### Por fim, no arquivo func.py há todas as funções realizadas no trabalho de 2020.

  Acredito que várias funções da mesma família deveriam ser agrupadas na mesma classe. É provável que há várias estruturas que não precisariam ser replicadas, e super().method. Habilitar @getter para facilitar no andamento do projeto.
  Falta aplicar Docstrings em todas as funções também. 
  Vários nomes estão horríveis, podem ser simplificados e ficarem mais claros.
  Exceções devem ser específicas, nunca gerais.
  
