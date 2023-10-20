# Updating the model parameters
In federated learning, the server sends the global model parameters to the client, and then the client updates the local model with the parameters received from the server. It then trains the model on the local data and sends the updated/changed model parameters back to the server (or could even send just the gradients back to server, and not even the full gradients).

## Implementing Flwr Client
- Federated systems consist of a server and multiple clients. In Flower we create clients by implementing subclasses of `flwr.client.Client` or `flwr.client.NumPyClient`.
- To implement Flower client, we create a subclass of `flwr.client.NumPyClient` and implement three methods:
    - `get_parameters`: Return the current local model parameters.
    - `fit`: Receive model parameters from the server, train the model parameters on the local data, and then return the updated model parameters to the server.
    - `evaluate`: Receive model parameters from the server, evaluate the model parameters on the local data and then return the metrics to server.
