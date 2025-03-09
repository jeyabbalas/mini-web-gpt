class Tensor {
    constructor(data, isShape = false) {
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
    }

    /**
     * Flatten nested arrays and determine shape
     * @param {Array} arr - Nested array
     * @returns {Object} Object with flat array and shape
     */
    flattenAndGetShape(arr) {
        const flat = [];
        const shape = [];

        function flatten(subArr, level) {
            if (!Array.isArray(subArr)) {
                flat.push(subArr);
                return;
            }

            if (shape.length <= level) {
                shape.push(subArr.length);
            } else if (shape[level] !== subArr.length) {
                throw new Error("Inconsistent shape in nested array.");
            }

            for (let i = 0; i < subArr.length; i++) {
                flatten(subArr[i], level + 1);
            }
        }

        flatten(arr, 0);

        return {flat, shape};
    }

    /**
     * Recursively rebuild a nested array from the flat Float32Array based on the given shape
     * @param {Float32Array} data - Flattened data
     * @param {Array} shape - Tensor shape
     * @returns {Array} - Reshaped nested array
     */
    unflatten(data, shape) {
        if (shape.length === 1) return Array.from(data);

        const size = shape[0];
        const subSize = data.length / size;
        let result = [];

        for (let i = 0; i < size; i++) {
            result.push(this.unflatten(data.slice(i * subSize, (i + 1) * subSize), shape.slice(1)));
        }

        return result;
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
}

export {Tensor};