# IUM (ML Engineering) project

Authors:
- Tomasz Owienko
- Anna Schäfer

---

Recommender system for generating music playlists for fictional "Pozytywka" service. Based on [LightFM](https://making.lyst.com/lightfm/docs/lightfm.html) algorithm.

Check out the [example notebook](src/example.ipynb).

### Project structure
- `data` - subsequent data versions for training the model and building the service
- `doc` - project documentation

### Run the service

Run with docker:
```shell
docker build -t ium .
docker run -p 8081:8081 -v ${PWD}:/predictions ium
```

All recommendations along with provided input and saved metadata will be saved to `/predictions/predictions.jsonl`.

**WARNING**: Bear in mind that build context and unused volume may (will) occupy large amounts of space, as docker will need to transfer the entire `data` directory to build daemon. A `docker system prune` may come in handy if you run out of space.

### Make recommendations

Following endpoints are available:
- `/predict` - make recommendations either with base or advanced model (A/B test). Exact model is chosen based on supplied user ID.
- `/predict/base` - predict using base model - sample songs from users' session history
- `/predict/advanced` - predict using the FM model

For exact request and response schemas check Swagger documentation at `/doc`.

