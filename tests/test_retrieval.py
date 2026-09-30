import numpy as np

from ragdetect.retrieval import NeighbourIndex, RetrievalDetector, neighbour_vote


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_index_matches_brute_force_cosine_search():
    rng = np.random.default_rng(0)
    pool, queries = rng.normal(size=(200, 16)), rng.normal(size=(10, 16))

    similarities, positions = NeighbourIndex(pool).search(queries, 5)

    brute = unit(queries) @ unit(pool).T
    expected = np.argsort(-brute, axis=1)[:, :5]
    np.testing.assert_array_equal(positions, expected)
    np.testing.assert_allclose(similarities, np.take_along_axis(brute, expected, axis=1), atol=1e-5)


def test_index_leaves_the_callers_arrays_alone():
    pool = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    NeighbourIndex(pool).search(pool, 1)
    np.testing.assert_array_equal(pool, [[3.0, 4.0], [1.0, 0.0]])


def test_k_larger_than_the_pool_is_clipped():
    _, positions = NeighbourIndex(np.eye(3)).search(np.eye(3), k=10)
    assert positions.shape == (3, 3)
    assert (positions >= 0).all()


def test_plain_vote_is_the_share_of_machine_neighbours():
    labels = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    np.testing.assert_allclose(neighbour_vote(np.zeros((2, 4)), labels), [0.5, 1.0])


def test_temperature_one_matches_the_original_experiment():
    rng = np.random.default_rng(1)
    scores = np.sort(rng.uniform(0.2, 0.9, size=(5, 7)), axis=1)[:, ::-1]
    labels = rng.integers(0, 2, size=(5, 7))

    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    original = (weights * labels).sum(axis=1) / weights.sum(axis=1)

    np.testing.assert_allclose(neighbour_vote(scores, labels, temperature=1.0), original)


def test_low_temperature_follows_the_nearest_neighbour():
    scores = np.array([[0.95, 0.60, 0.59, 0.58]])
    labels = np.array([[0, 1, 1, 1]])
    assert neighbour_vote(scores, labels, temperature=0.01)[0] < 0.01
    assert neighbour_vote(scores, labels)[0] == 0.75


def test_detector_separates_two_clusters():
    rng = np.random.default_rng(2)
    centre = rng.normal(size=16)
    pool = np.vstack([centre + 0.1 * rng.normal(size=(50, 16)), -centre + 0.1 * rng.normal(size=(50, 16))])
    labels = np.array([0] * 50 + [1] * 50)
    queries = np.vstack([centre, -centre])

    proba = RetrievalDetector(k=5).fit(pool, labels).predict_proba(queries)

    np.testing.assert_allclose(proba, [0.0, 1.0])
