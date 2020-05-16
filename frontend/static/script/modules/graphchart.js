class GraphChart{
    constructor(d3Svg){
        this.d3Svg = d3Svg;
        this.width = +this.d3Svg.attr('width');
        this.height = +this.d3Svg.attr('height');
        this.config = {
            padding: {left: 20, right: 20, bottom: 20, top: 20, row: 40, col: 40},
            nodeRadius: 4,
            nodeStrokeWidth: 2,
            edgeStrokeWidth: 2,
        };

        this.graphGroup = this.d3Svg.append('g').attr('class', 'graph-group')
            .attr('transform', `translate(${[this.config.padding.left, this.config.padding.top]})`);
    }

    fnDraw(nodes, edges){
        /**
         * @param nodes: a list of nodes position
         * @param edges: a list of edges. [[src, dst], ...]
         */

        let thisIns = this;
        let [xmin, xmax] = d3.extent(nodes.map(x => x[0]));
        let [ymin, ymax] = d3.extent(nodes.map(x => x[1]));
        console.log(xmin, xmax, ymin, ymax);
        console.log(this.config, this.width, this.height);
        let transformerx = new Transformer(xmin, xmax, this.config.padding.left, this.width - this.config.padding.row);
        let transformery = new Transformer(ymin, ymax, this.config.padding.top, this.height - this.config.padding.col);
        console.log(transformerx.fnTransform(xmin), this.config.padding.left);
        console.log(transformerx.fnTransform(xmax), this.width - this.config.padding.right);

        let nodesPos = nodes.map(item => [transformerx.fnTransform(item[0]), transformery.fnTransform(item[1])]);

        let edgesUpdate = this.graphGroup.selectAll('line.edge')
            .data(edges);
        edgesUpdate.exit().remove();
        edgesUpdate.enter()
            .append('line')
            .attr('class', 'edge')
            .attr('x1', function(edge){
                return nodesPos[edge[0]][0];
            })
            .attr('y1', function(edge){
                return nodesPos[edge[0]][1];
            })
            .attr('x2', function(edge){
                return nodesPos[edge[1]][0];
            })
            .attr('y2', function(edge){
                return nodesPos[edge[1]][1];
            })
            .style('stroke', 'grey')
            .style('stroke-width', this.config.edgeStrokeWidth);
        edgesUpdate.attr('x1', function(edge){
                return nodesPos[edge[0]][0];
            })
            .attr('y1', function(edge){
                return nodesPos[edge[0]][1];
            })
            .attr('x2', function(edge){
                return nodesPos[edge[1]][0];
            })
            .attr('y2', function(edge){
                return nodesPos[edge[1]][1];
            })
            .style('stroke', 'grey')
            .style('stroke-width', this.config.edgeStrokeWidth);

        let nodesUpdate = this.graphGroup.selectAll('circle.node')
            .data(nodesPos);
        nodesUpdate.exit().remove();
        nodesUpdate.enter()
            .append('circle')
            .attr('class', 'node')
            .attr('cx', function(node){
                return node[0];
            })
            .attr('cy', function(node){
                return node[1];
            })
            .attr('r', thisIns.config.nodeRadius)
            .style('fill', 'none')
            .style('stroke', 'black')
            .style('stroke-width', thisIns.config.nodeStrokeWidth);

        nodesUpdate.attr('cx', function(node){
                return node[0];
            })
            .attr('cy', function(node){
                return node[1];
            })
            .attr('r', thisIns.config.nodeRadius)
            .style('fill', 'none')
            .style('stroke', 'black')
            .style('stroke-width', thisIns.config.nodeStrokeWidth);
    }
}


class Transformer{
    constructor(orig_min, orig_max, target_min, target_max){
        this.k = (target_min - target_max) / (orig_min - orig_max);
        this.b = -(target_min * orig_max - target_max * orig_min) / (orig_min - orig_max);
    }

    fnTransform(x){
        return this.k * x + this.b;
    }
}