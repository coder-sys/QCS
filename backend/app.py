# app.py
from flask import Flask, request, jsonify
import cirq
import numpy as np
from werkzeug.exceptions import BadRequest
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CirqSimulator:
    def __init__(self):
        self.simulator = cirq.Simulator()

    def run_experiment(self, circuit_json):
        try:
            circuit = self._deserialize_circuit(circuit_json)
            result = self.simulator.simulate(circuit)

            # Convert complex numbers and numpy arrays to serializable formats
            state_vector = []
            for amplitude in result.state_vector():
                state_vector.append({
                    'real': float(amplitude.real),
                    'imag': float(amplitude.imag)
                })

            measurements = {}
            if hasattr(result, 'measurements'):
                for key, value in result.measurements.items():
                    measurements[key] = value.tolist()
            print(measurements,'here')
            return {
                'state_vector': state_vector,
                'measurements': measurements
            }

        except Exception as e:
            raise ValueError(f"Error simulating circuit: {str(e)}")

    def _deserialize_circuit(self, circuit_json):
        qubits = [cirq.GridQubit(*q) for q in circuit_json.get('qubits', [[0, 0]])]
        circuit = cirq.Circuit()

        for op in circuit_json.get('operations', []):
            gate_type = op['gate']
            targets = [qubits[i] for i in op['targets']]

            if gate_type == 'H':
                circuit.append(cirq.H.on_each(*targets))
            elif gate_type == 'X':
                circuit.append(cirq.X.on_each(*targets))
            elif gate_type == 'Y':
                circuit.append(cirq.Y.on_each(*targets))
            elif gate_type == 'Z':
                circuit.append(cirq.Z.on_each(*targets))
            elif gate_type == 'CNOT':
                circuit.append(cirq.CNOT(targets[0], targets[1]))
            elif gate_type == 'SWAP':
                circuit.append(cirq.SWAP(targets[0], targets[1]))
            elif gate_type == 'RX':
                circuit.append(cirq.rx(op['angle']).on_each(*targets))
            elif gate_type == 'RY':
                circuit.append(cirq.ry(op['angle']).on_each(*targets))
            elif gate_type == 'RZ':
                circuit.append(cirq.rz(op['angle']).on_each(*targets))
            elif gate_type == 'CCNOT':
                circuit.append(cirq.TOFFOLI(targets[0], targets[1], targets[2]))
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

        return circuit

simulator = CirqSimulator()

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
        {'name': 'MEASURE', 'description': 'Measurement'}
    ]
    return jsonify({'gates': gates})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')