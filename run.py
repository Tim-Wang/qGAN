import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

QiskitRuntimeService.save_account("", overwrite=True)
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print(f"Using backend: {backend}")

num_qubits = 3
qc = QuantumCircuit(num_qubits)
qc.h(qc.qubits)
ansatz = TwoLocal(num_qubits, "ry", "cz", entanglement="circular", reps=1)
qc.compose(ansatz, inplace=True)

weights = [-2.7978988, 0.14916761, 2.5675313, -1.9092216, -2.716498, 2.6611066]
# weights = [-8.8359942e-05, -6.9830291e-02,  1.6888468e+00, -4.1724644e+00, -3.6494882e+00,  2.4690874e+00, -2.5724576e+00, -5.3089827e-01, 2.4943171e+00]
bound_qc = qc.assign_parameters(weights)
bound_qc.measure_all()

fig_circuit = bound_qc.decompose().draw("mpl", scale=0.7, fold=100)
fig_circuit.tight_layout()
plt.show()

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_qc = pm.run(bound_qc)
sampler = Sampler(mode=backend)

print("Submitting job to backend...")
job = sampler.run([isa_qc], shots=2048)
print(f"Job ID: {job.job_id()}")

result = job.result()
pub_result = result[0]

counts = pub_result.data.meas.get_counts()
fig = plot_histogram(counts)
plt.show()