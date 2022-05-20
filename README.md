# Disengaged-Responses-in-Konwledge-Tracing
This repo includes the code of the poster presentation accepted at AIED 2022. 

###Summary of the study
We analyzed the influence of disengaged responses on the prediction accuracy of the knowledge tracing models (i.e., Bayesian Knowledge Tracing [BKT] and Deep Knowledge Tracing [DKT]). In this study, we created two sets of training and test data based on the data preprocessing stage. First, a baseline training set is created based by removing negative response times and empty skill names. We also created as disengaged-adjusted baseline model by removing disengaged responses in addition to the above preprocessing steps. We trained BKT and DKT and evaluated the model performance using both baseline and disengagement-adjusted models. The disengagement adjusted model outperformed the baseline model. 

####Resources:
- Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. Advances in neural information processing systems, 28.
- Pardos, Z. A., & Heffernan, N. T. (2010, June). Modeling individualization in a bayesian networks implementation of knowledge tracing. In International conference on user modeling, adaptation, and personalization (pp. 255-266). Springer, Berlin, Heidelberg.
- Feng, M., Heffernan, N., & Koedinger, K. (2009). Addressing the assessment challenge with an online system that tutors as it assesses. User modeling and user-adapted interaction, 19(3), 243-266.
- https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010
