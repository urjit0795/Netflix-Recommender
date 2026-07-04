# Evaluation Strategy

Recommendation systems should be evaluated differently from classification models.

## Offline Metrics

- Precision@K: How many recommended movies are relevant?
- Recall@K: How many relevant movies did we retrieve?
- MAP@K: How well are relevant items ranked?
- NDCG@K: Are highly relevant movies ranked near the top?
- Coverage: How much of the catalog can the model recommend?
- Diversity: Are recommendations too similar?

## Online Metrics

In production, I would evaluate:

- Click-through rate
- Watch time
- Add-to-list rate
- User retention
- Session engagement
- A/B test lift