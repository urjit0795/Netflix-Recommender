# Production Considerations

If this recommendation system were deployed in production, I would consider:

## Serving
- Expose recommendations using a FastAPI endpoint
- Cache popular movie recommendations
- Precompute recommendations for active users

## Evaluation
- Track offline metrics such as Precision@K, Recall@K, MAP@K and NDCG@K
- Run A/B tests to measure engagement lift
- Monitor cold-start performance for new users and new movies

## Monitoring
- Track recommendation diversity and freshness
- Monitor data drift in user behavior
- Detect popularity bias
- Measure latency and system reliability

## Future Improvements
- Add hybrid ranking
- Add contextual features
- Include real-time user behavior
- Build a feedback loop from user interactions