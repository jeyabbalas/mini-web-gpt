class Tensor {
    constructor(data, isShape = false) {
        if (isShape) {
            const shape = Array.isArray(data) ? data : [data];
            const size = shape.reduce((a, b) => a * b, 1);
            this.data = new Float32Array(size);
            this.shape = shape;
        }
    }

    flattenAndGetShape(arr) {
    }

    toString() {
        if (this.shape.length === 1) {
            return `tensor(${JSON.stringify(Array.from(this.data))}, dtype=float32)`;
        }

        const formattedArray = this.unflatten(this.data, this.shape);
        return `tensor(${JSON.stringify(formattedArray).replace(/,/g, ', ')}, dtype=float32)`;
    }

    /**
     * Convert a flat Float32Array back into a nested array based on the tensor shape
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
}

export {Tensor};