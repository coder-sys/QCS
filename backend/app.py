from flask import Flask, request, jsonify
import cirq
import numpy as np
from werkzeug.exceptions import BadRequest
from flask_cors import CORS
import json
import time
from collections import defaultdict
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv('env')

class EnhancedCirqSimulator:
    def __init__(self):
        self.simulator = cirq.Simulator()
        self.circuit_cache = {}
        self.result_cache = {}
    
    def _generate_circuit_id(self, circuit_json):
        """Generate a unique ID for the circuit based on its content"""
        return hash(json.dumps(circuit_json, sort_keys=True))
    
    def run_experiment(self, circuit_json):
        try:
            circuit_id = self._generate_circuit_id(circuit_json)
            
            # Check cache first
            if circuit_id in self.result_cache:
                return self.result_cache[circuit_id]
                
            circuit = self._deserialize_circuit(circuit_json)
            start_time = time.time()
            result = self.simulator.simulate(circuit)
            execution_time = time.time() - start_time
            
            # Convert results to serializable format
            processed_result = self._process_result(result)
            processed_result['execution_time'] = execution_time
            processed_result['qubit_count'] = len(circuit.all_qubits())
            processed_result['gate_count'] = len(list(circuit.all_operations()))
            
            # Cache the result
            self.result_cache[circuit_id] = processed_result
            return processed_result

        except Exception as e:
            raise ValueError(f"Error simulating circuit: {str(e)}")
    
    def _process_result(self, result):
        """Convert cirq result to serializable format"""
        state_vector = []
        for amplitude in result.state_vector():
            state_vector.append({
                'real': float(amplitude.real),
                'imag': float(amplitude.imag)
            })

        measurements = defaultdict(list)
        if hasattr(result, 'measurements'):
            for key, value in result.measurements.items():
                measurements[key] = value.tolist()

        return {
            'state_vector': state_vector,
            'measurements': dict(measurements)
        }
    
    def _qft(self, qubits):
        """Quantum Fourier Transform on the given qubits."""
        qreg = list(qubits)
        for i in range(len(qreg)):
            yield cirq.H(qreg[i])
            for j in range(1, len(qreg) - i):
                yield (cirq.CZ ** (1/(2 ** j)))(qreg[i], qreg[i + j])
                yield cirq.rz(np.pi/(2 ** j)).on(qreg[i + j])
        # Swap the qubits to complete QFT
        for i in range(len(qreg) // 2):
            yield cirq.SWAP(qreg[i], qreg[len(qreg) - i - 1])
    
    def _deserialize_circuit(self, circuit_json):
        """Convert JSON circuit description to Cirq circuit"""
        circuit_id = self._generate_circuit_id(circuit_json)
        
        # Check cache first
        if circuit_id in self.circuit_cache:
            return self.circuit_cache[circuit_id]
            
        qubits = [cirq.GridQubit(*q) for q in circuit_json.get('qubits', [[0, i] for i in range(len(circuit_json.get('operations', [])))])]
        circuit = cirq.Circuit()

        for op in circuit_json.get('operations', []):
            gate_type = op['gate']
            targets = [qubits[i] for i in op['targets']]
            controls = [qubits[i] for i in op.get('controls', [])]

            if gate_type == 'H':
                circuit.append(cirq.H.on_each(*targets))
            elif gate_type == 'X':
                if controls:
                    circuit.append(cirq.X.on(targets[0]).controlled_by(*controls))
                else:
                    circuit.append(cirq.X.on_each(*targets))
            elif gate_type == 'Y':
                if controls:
                    circuit.append(cirq.Y.on(targets[0]).controlled_by(*controls))
                else:
                    circuit.append(cirq.Y.on_each(*targets))
            elif gate_type == 'Z':
                if controls:
                    circuit.append(cirq.Z.on(targets[0]).controlled_by(*controls))
                else:
                    circuit.append(cirq.Z.on_each(*targets))
            elif gate_type == 'CNOT':
                if not controls and len(targets) < 2:
                    raise ValueError("CNOT gate requires either controls or two targets")
                control = controls[0] if controls else targets[0]
                target = targets[0] if controls else targets[1]
                if control == target:
                    raise ValueError("Control and target qubits must be different")
                circuit.append(cirq.CNOT(control, target))
            elif gate_type == 'SWAP':
                circuit.append(cirq.SWAP(targets[0], targets[1]))
            elif gate_type == 'RX':
                circuit.append(cirq.rx(op['angle']).on_each(*targets))
            elif gate_type == 'RY':
                circuit.append(cirq.ry(op['angle']).on_each(*targets))
            elif gate_type == 'RZ':
                circuit.append(cirq.rz(op['angle']).on_each(*targets))
            elif gate_type == 'CCNOT':
                if len(controls) >= 2 and len(targets) >= 1:
                    # Ensure all qubits are unique
                    all_qubits = controls + targets[:1]
                    if len(set(all_qubits)) != len(all_qubits):
                        raise ValueError("CCNOT gate requires unique qubits for controls and target")
                    circuit.append(cirq.TOFFOLI(controls[0], controls[1], targets[0]))
                else:
                    raise ValueError("CCNOT/Toffoli gate requires exactly two controls and one target")
            elif gate_type == 'S':
                circuit.append(cirq.S.on_each(*targets))
            elif gate_type == 'T':
                circuit.append(cirq.T.on_each(*targets))
            elif gate_type == 'I':
                circuit.append(cirq.I.on_each(*targets))
            elif gate_type == 'CPHASE':
                circuit.append(cirq.CZ(targets[0], targets[1]))
            elif gate_type == 'Measure':
                circuit.append(cirq.measure(*targets, key=op.get('key', 'result')))
            elif gate_type == 'QFT':
                circuit.append(self._qft(targets))

        # Cache the circuit
        self.circuit_cache[circuit_id] = circuit
        return circuit

simulator = EnhancedCirqSimulator()

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.get_json()
        if not data or 'circuit' not in data:
            raise BadRequest("Circuit data is required")

        result = simulator.run_experiment(data['circuit'])
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/quantum-assistant', methods=['POST'])
def quantum_assistant():
    try:
        if not openai.api_key:
            return jsonify({'success': False, 'error': 'OpenAI API key not configured'}), 500

        data = request.get_json()
        if not data or 'messages' not in data or 'qubit_count' not in data:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        system_msg = {
            "role": "system",
            "content": f"""You are a quantum computing assistant specialized in circuit construction. Follow these rules:
1. Always respond with a clear explanation first
2. For circuit modifications, include a JSON array after '||' with this exact format:
   [{{"gate": "GATE_TYPE", "targets": [qubit_index], "controls": [optional_control_indices], "angle": optional_angle}}]
   
Available gates: H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CCNOT, S, T, CPHASE, Measure, QFT
Current qubits: {data['qubit_count']}
Qubit indices: 0 to {data['qubit_count']-1}

Example valid responses:
"To create superposition, apply Hadamard to qubit 0. || [{{\"gate\": \"H\", \"targets\": [0]}}]"
"Entangle qubits 0 and 1 with CNOT (0 controls 1) || [{{\"gate\": \"CNOT\", \"targets\": [1], \"controls\": [0]}}]"
"Rotate qubit 0 by π/2 around X-axis || [{{\"gate\": \"RX\", \"targets\": [0], \"angle\": 1.5708}}]"
"""
        }

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[system_msg] + data['messages'],
            temperature=0.3
        )

        full_response = response.choices[0].message.content
        if '||' in full_response:
            explanation, commands_str = full_response.split('||', 1)
            try:
                commands = json.loads(commands_str.strip())
                valid_commands = []
                for cmd in commands:
                    if not all(k in cmd for k in ['gate', 'targets']):
                        continue
                    if cmd['gate'] not in ['H', 'X', 'Y', 'Z', 'CNOT', 'SWAP', 'RX', 'RY', 'RZ', 'CCNOT', 'S', 'T', 'CPHASE', 'Measure', 'QFT']:
                        continue
                    if any(t >= data['qubit_count'] for t in cmd['targets']):
                        continue
                    if 'controls' in cmd and any(c >= data['qubit_count'] for c in cmd['controls']):
                        continue
                    valid_commands.append(cmd)
            except json.JSONDecodeError:
                valid_commands = []
        else:
            explanation = full_response
            valid_commands = []

        return jsonify({
            'success': True,
            'explanation': explanation.strip(),
            'commands': valid_commands
        })

    except openai.error.AuthenticationError:
        return jsonify({'success': False, 'error': 'Invalid OpenAI API key'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gates', methods=['GET'])
def available_gates():
    gates = [
        {'name': 'H', 'description': 'Hadamard gate'},
        {'name': 'X', 'description': 'Pauli-X gate'},
        {'name': 'Y', 'description': 'Pauli-Y gate'},
        {'name': 'Z', 'description': 'Pauli-Z gate'},
        {'name': 'CNOT', 'description': 'Controlled-NOT gate'},
        {'name': 'SWAP', 'description': 'SWAP gate'},
        {'name': 'RX', 'description': 'X-axis rotation'},
        {'name': 'RY', 'description': 'Y-axis rotation'},
        {'name': 'RZ', 'description': 'Z-axis rotation'},
        {'name': 'CCNOT', 'description': 'Toffoli gate'},
        {'name': 'S', 'description': 'Phase gate'},
        {'name': 'T', 'description': 'T gate'},
        {'name': 'I', 'description': 'Identity gate'},
        {'name': 'CPHASE', 'description': 'Controlled Phase gate'},
        {'name': 'MEASURE', 'description': 'Measurement'},
        {'name': 'QFT', 'description': 'Quantum Fourier Transform'}
    ]
    return jsonify({'gates': gates})

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    simulator.circuit_cache.clear()
    simulator.result_cache.clear()
    return jsonify({'success': True, 'message': 'Cache cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
