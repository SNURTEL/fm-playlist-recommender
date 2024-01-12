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
docker run -p 8000:8000 ium
```

**WARNING**: Bear in mind that build context and unused volume may (will) occupy large amounts of space, as docker will need to transfer the entire `data` directory to build daemon and then copy it to volume. A `docker system prune` and `docker volume prune -a` may come in handy if you run out of space.