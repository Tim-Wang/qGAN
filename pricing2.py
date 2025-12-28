import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, LinearAmplitudeFunction
from qiskit.primitives import StatevectorSampler 
from qiskit_algorithms import EstimationProblem, MaximumLikelihoodAmplitudeEstimation
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ==========================================
# 连接 IBM Quantum 真机
# ==========================================
QiskitRuntimeService.save_account("", overwrite=True)
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print(f"Using backend: {backend}")

# ==========================================
# 自动 Transpile 的 Sampler 包装器
# ==========================================
class TranspilingSampler:
    """包装 IBM Sampler，自动 transpile 电路"""
    def __init__(self, sampler, pass_manager):
        self._sampler = sampler
        self._pm = pass_manager
    
    @property
    def default_shots(self):
        return getattr(self._sampler, 'default_shots', 4096)
    
    def run(self, pubs, **kwargs):
        transpiled_pubs = []
        for pub in pubs:
            if isinstance(pub, tuple):
                circuit = pub[0]
                rest = pub[1:]
                transpiled_circuit = self._pm.run(circuit)
                transpiled_pubs.append((transpiled_circuit,) + rest)
            else:
                transpiled_pubs.append(self._pm.run(pub))
        return self._sampler.run(transpiled_pubs, **kwargs)

# ==========================================
# 第一部分：定义概率加载电路 (P_X)
# ==========================================
num_uncertainty_qubits = 3
qc = QuantumCircuit(num_uncertainty_qubits)
qc.h(qc.qubits)
ansatz = TwoLocal(num_uncertainty_qubits, "ry", "cz", entanglement="circular", reps=1)
qc.compose(ansatz, inplace=True)

weights = [-2.7978988, 0.14916761, 2.5675313, -1.9092216, -2.716498, 2.6611066]
bound_qc = qc.assign_parameters(weights)

# ==========================================
# 第二部分：构建期权定价算法 (MLAE)
# ==========================================
S_min, S_max = 0, 7
strike_price_K = 2
c_approx = 0.25 

# 定义收益函数 (参考 EuropeanCallPricingObjective 的正确实现)
payoff_func = LinearAmplitudeFunction(
    num_state_qubits=num_uncertainty_qubits,
    slope=[0, 1],                
    offset=[0, 0],
    domain=(S_min, S_max),
    image=(0, S_max - strike_price_K),
    rescaling_factor=c_approx,
    breakpoints=[S_min, strike_price_K] 
)

# 组合算子 A
num_qubits = payoff_func.num_qubits
state_preparation = QuantumCircuit(num_qubits)
state_preparation.append(bound_qc, range(num_uncertainty_qubits))
state_preparation.append(payoff_func, range(num_qubits))

objective_qubits = [num_uncertainty_qubits]

problem = EstimationProblem(
    state_preparation=state_preparation,
    objective_qubits=objective_qubits,
    post_processing=payoff_func.post_processing
)

print(f"Total qubits: {state_preparation.num_qubits}")
print(f"Objective qubit index: {objective_qubits[0]}")

# ==========================================
# 分析电路的门深度和门数量
# ==========================================
def analyze_circuit(circuit, name="Circuit", backend=None, pm=None):
    """分析电路的门深度、门数量等信息"""
    # 先分解电路以获得基本门
    decomposed = circuit.decompose().decompose()
    
    print(f"\n--- {name} 分析 (分解后) ---")
    print(f"量子比特数: {decomposed.num_qubits}")
    print(f"电路深度 (depth): {decomposed.depth()}")
    print(f"总门数量 (size): {decomposed.size()}")
    print(f"门统计: {dict(decomposed.count_ops())}")
    
    # 如果提供了 backend 和 pass_manager，分析 transpile 后的电路
    if backend is not None and pm is not None:
        transpiled = pm.run(circuit)
        print(f"\n--- {name} 分析 (Transpile 后 @ {backend.name}) ---")
        print(f"电路深度 (depth): {transpiled.depth()}")
        print(f"总门数量 (size): {transpiled.size()}")
        print(f"门统计: {dict(transpiled.count_ops())}")
        # 双量子比特门数量（通常是 ECR 或 CX）
        two_qubit_gates = sum(count for gate, count in transpiled.count_ops().items() 
                             if gate in ['cx', 'ecr', 'cz', 'swap', 'iswap'])
        print(f"双量子比特门数量: {two_qubit_gates}")

# 使用 IBM Runtime Sampler (真机) + 自动 Transpile
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

# 分析 state_preparation 电路 (这是 A 算子)
analyze_circuit(state_preparation, "State Preparation (A算子)", backend, pm)

print("\n" + "="*50)
print("Constructing MLAE...")

raw_sampler = Sampler(mode=backend)
sampler = TranspilingSampler(raw_sampler, pm)
# sampler = StatevectorSampler()

mlae = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=[0, 1, 2, 3], 
    sampler=sampler
)

print("Running Estimation on real quantum hardware...")
print(f"Job submitted to {backend.name}")
result = mlae.estimate(problem)

# 结果后处理 - 使用 estimation_processed 获取已经通过 post_processing 处理的结果
estimated_price = result.estimation_processed

print(f"\n--- Pricing Results (Real Hardware) ---")
print(f"Backend: {backend.name}")
print(f"Estimated Amplitude: {result.estimation:.6f}")
print(f"Estimated Option Price: {estimated_price:.6f}")