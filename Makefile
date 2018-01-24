.PHONY: cluster, helm, dask

cluster:
	gcloud container clusters create dask-demo \
		--num-nodes=3 \
		--machine-type=n1-standard-2 \
		--zone=europe-west1-b

helm:
	kubectl --namespace kube-system create sa tiller && \
	kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller && \
	helm init --service-account tiller && \
	helm repo add dask https://dask.github.io/helm-chart && \
	helm repo update

dask:
	helm install -f config.yaml dask/dask
