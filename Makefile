.PHONY: cluster

cluster:
	dask-kubernetes create distributed-joblib-benchmark settings.yaml
