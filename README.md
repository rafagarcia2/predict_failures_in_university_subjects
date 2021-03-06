# Predição de Reprovações na disciplina de Cálculo II da ECT/UFRN.
Esse projeto tem como objetivo utilizar os dados disponíveis no portal de dados abertos da UFRN para realizar predições sobre as reprovações dos discentes matriculados na disciplina de Cálculo II da Escola de Ciência e Técnologia (ECT) - UFRN.

 
#### Grupo:
- Rafael Garcia Dantas
- Mateus Eloi Bastos
 
### Introdução:
 
Uma das áreas que encontramos uma vasta gama de possibilidades para aplicação de conceitos de aprendizado de máquinas e ciência de dados é a área educacional. Desde previsão da nota de um aluno até recomendação de quais disciplinas o mesmo deveria pagar. São inúmeras as problemáticas que encontramos, nos possibilitando aplicar modelos inteligentes em nossas soluções.
 
Inspirados nesse leque de possibilidades, resolvemos escolher uma problemática bem interessante no contexto educacional. A "previsão" de desempenho dos alunos em determinadas disciplinas baseado em seu histórico acadêmico.
Atuamos primariamente dentro da UFRN, mais especificamente observando os alunos do Bacharelado em ciências e tecnologia.
 
### Metodologia:
 
Primeiramente definimos um grupo amostral específico e estabelecemos algumas premissas: Escolhemos observar o comportamento dos alunos de C&T em seus semestres iniciais, mais especificamente as disciplinas de exatas. Através de uma análise exploratória dos dados ( dados esses carregados do portal de dados abertos da UFRN) percebemos uma forte correlação entre o desempenho nas disciplinas de exatas do primeiro semestre, tais como: Vetores e Geometria analítica (VGA), Pré Cálculo e Cálculo I. Com as disciplina de exatas do segundo semestre.Observando esse fato, surge a pergunta: porque não criar um modelo que consiga prever o sucesso ou fracasso do aluno com base no seu desempenho nas disciplinas citadas acima? Com a hipótese levantada, começamos o processo de coleta de dados, transformação, elaboração de novas características, treinamento do modelo de aprendizado de máquina, validação dos resultados e por fim o intuito final da disciplina, colocar toda essa estrutura em produção.


### Ferramentas

Para desenvolver esse projeto utilizamos de várias ferramentas da área de Data Science e MLOps. Tendo como base a linguagem Python, utilizamos bibliotecas como:
- Pandas
- Numby
- Sklearn
- Matplotlib
- Entre outras

Para construir o pipeline e subir o modelo para produção, foram utilizadas várias ferramentas de MLOps, entre elas:
- MLflow
- Weights and Bias
- Hydra
- Pytest
- Conda


### Reproduzindo o pipelina:
Para facilitar o entendimento e permitir a reprodução do pipeline, disponibilizamos alguns tutoriais demonstrando como executar o projeto corretamente:
1. [Entendendo a problemática](https://www.loom.com/share/8da8a47ba75b40d8a7e8b6ca8670e43b)
2. [Pipeline de reprodução e MLflow](https://www.loom.com/share/e63bdb14b3ff4154992c40bc24ef7b1b)
3. [Download e Pré-processamento](https://www.loom.com/share/59f83f52cba3479bb3d4144a3ca06c77)
4. [Testes e separação em train/test](https://www.loom.com/share/e97cfbad6fca4c44a8a2bd484b3da96e)
5. [Modelo e Análise de resultados](https://www.loom.com/share/603824da895d48f087452d1fe519cce2)


### Conclusão:
 
O aprendizado que conseguimos obter depois de construir todo esse pipeline de dados é gigantesco! Desde a elaboração da problemática até colocar o modelo em produção de uma forma reprodutível, escalável e dirigido a testes, com certeza agrega muito valor em nossa formação.
Abre-se também a possibilidade de tornar nosso problema mais complexo, adicionando mais disciplinas ao conjunto de treinamento e gerando ainda mais previsões para outras disciplinas em outras áreas. 







