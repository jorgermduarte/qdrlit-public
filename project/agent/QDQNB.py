import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector, Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator

class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.n_qubits = 1
        self.n_layers = 1
        self.rotation_blocks = ['rx', 'ry', 'rz']
        self.entanglement_blocks = 'cx'
        self.entanglement_type = 'linear' # 'linear' or 'full'
        
        num_input_features = self.n_qubits * len(self.rotation_blocks)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, num_input_features),
        )

        input_params = ParameterVector("x", num_input_features)
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(input_params[3*i], i)
            qc.ry(input_params[3*i + 1], i)
            qc.rz(input_params[3*i + 2], i)

        ansatz_unbound = TwoLocal(
            num_qubits=self.n_qubits,
            rotation_blocks=self.rotation_blocks,
            entanglement_blocks=self.entanglement_blocks,
            entanglement=self.entanglement_type,
            reps=self.n_layers,
            insert_barriers=False
        )

        weight_params_vector = ParameterVector("θ", len(ansatz_unbound.parameters))

        ansatz_bound = ansatz_unbound.assign_parameters(
            {q_param: weight_params_vector[i] for i, q_param in enumerate(ansatz_unbound.parameters)}
        )
        qc.compose(ansatz_bound, inplace=True)

        observables = [
            SparsePauliOp('I'*i + 'Z' + 'I'*(self.n_qubits - i - 1))
            for i in range(self.n_qubits)
        ]
        
        self.qnn = TorchConnector(
            EstimatorQNN(
                circuit=qc,
                input_params=input_params,
                weight_params=list(weight_params_vector),
                observables=observables,
                input_gradients=True,
            )
        )

        self.quantum_boost = nn.Sequential(
            nn.Linear(self.n_qubits, output_dim),
        )

        self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        projected = self.input_proj(x)
        q_out = self.qnn(projected)
        boosted = self.quantum_boost(q_out)
        print(f"q_out quantum norm: {torch.norm(boosted)}")
        return boosted

class QDQN(nn.Module):
    def __init__(self, input_size=45, output_size=90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            QuantumLayer(input_dim=256, output_dim=256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        output = self.net(x)
        if output.dim() == 2 and output.size(0) == 1:
            output = output.squeeze(0)
        return output
