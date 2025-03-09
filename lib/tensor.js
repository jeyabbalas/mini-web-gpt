class Tensor {
    /**
     * Create a new Tensor object
     * @param {Array|Float32Array|number} data - The data or shape
     * @param {boolean} isShape - Whether data represents shape (to create zeros tensor)
     */
    constructor(data, isShape = false) {
        if (!Tensor.isValidTensor(data)) {
            throw new Error('Invalid tensor data: inconsistent shape or unsupported data type.');
        }

        if (isShape) {
            const shape = Array.isArray(data) ? data : [data];
            const size = shape.reduce((a, b) => a * b, 1);
            this.data = new Float32Array(size);
            this.shape = shape;
        } else if (Array.isArray(data)) {
            const {flat, shape} = this.flattenAndGetShape(data);
            this.data = new Float32Array(flat);
            this.shape = shape;
        } else if (data instanceof Float32Array) {
            this.data = data;
            this.shape = [data.length];
        } else if (typeof data === 'number') {
            this.data = new Float32Array([data]);
            this.shape = [1];
        } else {
            throw new Error('Invalid data type for Tensor');
        }

        // Automatic differentiation
        this.requiresGrad = false; // Compute gradients with respect to this tensor?
        this.grad = null;          // Gradient of the loss with respect to this tensor
        this._parents = [];        // Tensors that this tensor depends on
        this._backward = null;     // Function to compute the gradient of the loss with respect to this tensor
    }

    /**
     * Validate whether an input is a valid tensor-like object.
     * Ensures:
     * - Data is a number, Float32Array, or a nested array of consistent shape.
     * - No mixed-length arrays.
     *
     * @param {any} data - The input tensor data to validate.
     * @returns {boolean} - True if valid, otherwise false.
     */
    static isValidTensor(data) {
        if (typeof data === 'number' || data instanceof Float32Array) {
            return true;
        }

        if (!Array.isArray(data)) return false;

        let shape = null;

        function checkShape(arr, depth = 0) {
            if (!Array.isArray(arr)) return true;

            if (shape === null) {
                shape = [];
            }

            if (shape.length <= depth) {
                shape.push(arr.length);
            } else if (shape[depth] !== arr.length) {
                return false; // Inconsistent shape detected
            }

            for (let i = 0; i < arr.length; i++) {
                if (!checkShape(arr[i], depth + 1)) {
                    return false;
                }
            }
            return true;
        }

        return checkShape(data);
    }

    /**
     * Flatten nested arrays and determine shape
     * @param {Array} arr - Nested array
     * @returns {Object} Object with flat array and shape
     */
    flattenAndGetShape(arr) {
        const flat = [];
        const shape = [];
        const stack = [{array: arr, depth: 0}];

        while (stack.length > 0) {
            const {array, depth} = stack.pop();

            if (!Array.isArray(array)) {
                flat.push(array);
                continue;
            }

            if (shape.length <= depth) {
                shape.push(array.length);
            } else if (shape[depth] !== array.length) {
                throw new Error("Inconsistent shape in nested array.");
            }

            for (let i = array.length - 1; i >= 0; i--) {
                stack.push({array: array[i], depth: depth + 1});
            }
        }

        return {flat, shape};
    }

    /**
     * Recursively rebuild a nested array from the flat Float32Array based on the given shape
     * @param {Float32Array} data - Flattened data
     * @param {Array} shape - Tensor shape
     * @param {number} [offset=0] - The starting index in the flattened data (used for recursion).
     * @returns {Array} - Reshaped nested array
     */
    unflatten(data, shape, offset = 0) {
        if (shape.length === 1) return Array.from(data.subarray(offset, offset + shape[0]));

        const size = shape[0];
        const subSize = data.length / size;
        let result = new Array(size);

        for (let i = 0; i < size; i++) {
            result[i] = this.unflatten(data, shape.slice(1), offset + i * subSize);
        }

        return result;
    }

    /**
     * Get the size of a dimension of the tensor
     * @param {number} dim
     * @returns {number} Size of the dimension
     */
    size(dim) {
        return this.shape[dim];
    }

    /**
     * Recursively format a nested array into a string that looks like a matrix.
     * @param {Array|number} nested - The nested array or a number
     * @param {number} indent - Current indentation level (number of spaces)
     * @returns {string} - The formatted string
     */
    formatTensorString(nested, indent = 0) {
        const indentStr = ' '.repeat(indent);
        if (!Array.isArray(nested)) {
            return nested.toString();
        }
        const is1D = !nested.some(element => Array.isArray(element));
        if (is1D) {
            return '[' + nested.join(', ') + ']';
        } else {
            const innerIndent = indent + 2;
            const innerIndentStr = ' '.repeat(innerIndent);
            const lines = nested.map(sub => innerIndentStr + this.formatTensorString(sub, innerIndent));
            return '[\n' + lines.join(',\n') + '\n' + indentStr + ']';
        }
    }

    /**
     * Convert tensor into a formatted string
     * @returns {string} - The tensor represented as a string
     */
    toString() {
        let formatted;
        if (this.shape.length === 1) {
            formatted = this.formatTensorString(Array.from(this.data));
        } else {
            const nested = this.unflatten(this.data, this.shape);
            formatted = this.formatTensorString(nested);
        }
        return `tensor(${formatted}, dtype=float32)`;
    }

    /**
     * Clone the tensor
     * @returns {Tensor} A new tensor with the copy of this tensor's data
     */
    clone() {
        const newTensor = new Tensor(this.shape, true);
        newTensor.data.set(this.data);
        return newTensor;
    }

    /**
     * Set gradients of the tensor to zero
     */
    zeroGrad() {
        if (this.grad) {
            this.grad.data.fill(0);
        }
    }
}

export {Tensor};