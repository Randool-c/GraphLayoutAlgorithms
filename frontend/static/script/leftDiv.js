class DatasetInfoDiv {
    constructor() {
        this.$datasetInfoTable = $('#dataset-info-div table');
        console.log(this.$datasetInfoTable);
    }

    fnUpdateTable(datasetInfo) {
        let importantKeys = ['Name', 'Num Rows', 'Nonzeros', 'Positive Definite',
            'Minimum Singular Value', 'Matrix Norm'];
        let keyDict = {
            'Num Rows': 'Node',
            'Nonzeros': 'Edge'
        };
        let rowsHTML = ``;
        for (let key of importantKeys) {
            let keyname;
            if (datasetInfo.hasOwnProperty(key)) {
                if (keyDict.hasOwnProperty(key)) {
                    keyname = keyDict[key];
                } else {
                    keyname = key;
                }
                rowsHTML += `<tr><td>${keyname}</td><td>${datasetInfo[key]}</td></tr>`;
            }
        }
        console.log(rowsHTML);
        this.$datasetInfoTable.html(`<tbody>${rowsHTML}</tbody>`);
    }

    fnUpdateError() {
        /*
            数据集请求错误
         */
        this.$datasetInfoTable.text('ERROR');
    }
}


class LayoutParamsDiv {
    constructor() {
        this.layoutParamsDiv = $('#layout-params-div');
        this.layoutMethod = 'none';
    }

    fnUpdate(layoutMethod) {
        let thisObj = this;
        if (this.layoutMethod !== layoutMethod) {
            this.layoutParamsDiv.slideUp("fast", function () {
                if (layoutMethod === 'sm') {
                    // params: stop_th, cg_th
                    thisObj.layoutParamsDiv.html(`
                        <div class="form-group row p-1">
                            <label for="stop-th-range" class="col-sm-4 col-form-label col-form-label-sm">stop th</label>
                            <input type="range" name="stop_th" class="custom-range col-sm-4 mt-2" min="1e-5" max="0.1" step="1e-5" id="stop-th-range" value="0.01"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="cg-th-range" class="col-sm-4 col-form-label col-form-label-sm">cg th</label>
                            <input type="range" name="cg_th" class="col-sm-4 custom-range mt-2" min="1e-5" max="0.1" step="1e-5" id="cg-th-range" value="0.1"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                    `).slideDown();
                } else if (layoutMethod === 'sgd') {
                    thisObj.layoutParamsDiv.html(`
                        <div class="form-group row p-1">
                            <label for="stop-th-range" class="col-sm-4 col-form-label col-form-label-sm">stop th</label>
                            <input type="range" name="stop_th" class="custom-range col-sm-4 mt-2" min="0.01" max="0.5" step="0.01" id="stop-th-range"
                                   value="0.03"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="tmax-range" class="col-sm-4 col-form-label col-form-label-sm">tmax</label>
                            <input type="range" name="tmax" class="col-sm-4 custom-range mt-2" min="1" max="50" step="1" id="tmax-range"
                                   value="30"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="tmaxmax-range" class="col-sm-4 col-form-label col-form-label-sm">tmaxmax</label>
                            <input type="range" name="tmaxmax" class="col-sm-4 custom-range mt-2" min="50" max="500" step="1" id="tmaxmax-range"
                                   value="200"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="eps-range" class="col-sm-4 col-form-label col-form-label-sm">eps</label>
                            <input type="range" name="eps" class="col-sm-4 custom-range mt-2" min="0.01" max="0.2" step="0.01" id="eps-range"
                                   value="0.03"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                    `).slideDown();
                }
                else if (layoutMethod === 'none'){
                    thisObj.layoutParamsDiv.html('');
                }
                thisObj.layoutParamsDiv.find('input[type=range]').mousemove(function () {
                    $(this).siblings('input[type=text]').val(this.value);
                }).mousemove();
                thisObj.layoutParamsDiv.find('input[type=text]').change(function () {
                    $(this).siblings('input[type=range]').val(parseFloat(this.value));
                });
            });
        }

        this.layoutMethod = layoutMethod;
    }

    fnGetParams(){
        let params = {};
        this.layoutParamsDiv.find('input[type=range]').forEach(function(){
            params[this.name] = this.value;
        });
        return params;
    }
}


class MultiscaleParamsDiv {
    constructor() {
        this.multiscaleParamsDiv = $('#multiscale-params-div');
        this.multiscaleMethod = 'none';
    }

    fnUpdate(multiscaleMethod) {
        let thisObj = this;
        if (this.multiscaleMethod !== multiscaleMethod) {
            this.multiscaleParamsDiv.slideUp("fast", function () {
                if (multiscaleMethod === 'fast' || multiscaleMethod === 'edge_contracting' || multiscaleMethod === 'adapted_init') {
                    thisObj.multiscaleParamsDiv.html(`
                        <div class="form-group row p-1">
                            <label for="ratio-range" class="col-sm-4 col-form-label col-form-label-sm">ratio</label>
                            <input type="range" name="ratio" class="col-sm-4 custom-range mt-2" min="0.1" max="0.9" step="0.1" id="ratio-range"
                                   value="0.5"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="th-range" class="col-sm-4 col-form-label col-form-label-sm">threshold</label>
                            <input type="range" name="th" class="col-sm-4 custom-range mt-2" min="20" max="200" step="1" id="th-range"
                                   value="100"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                    `).slideDown();
                } else if (multiscaleMethod === 'weighted_interpolation') {
                    thisObj.multiscaleParamsDiv.html(`
                        <div class="form-group row p-1">
                            <label for="t-range" class="col-sm-4 col-form-label col-form-label-sm">t</label>
                            <input type="range" name="t" class="col-sm-4 custom-range mt-2" min="0.01" max="0.1" step="0.01" id="t-range"
                                   value="0.05"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="delta-t-range" class="col-sm-4 col-form-label col-form-label-sm">delta_t</label>
                            <input type="range" name="delta_t" class="col-sm-4 custom-range mt-2" min="0.02" max="0.08" step="0.01" id="delta-t-range"
                                   value="0.05"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="sweeps-range" class="col-sm-4 col-form-label col-form-label-sm">num_sweeps</label>
                            <input type="range" name="num_sweeps" class="col-sm-4 custom-range mt-2" min="3" max="10" step="1" id="sweeps-range"
                                   value="5"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                        <div class="form-group row p-1">
                            <label for="th-range" class="col-sm-4 col-form-label col-form-label-sm">threshold</label>
                            <input type="range" name="th" class="col-sm-4 custom-range mt-2" min="20" max="200" step="1" id="th-range"
                                   value="100"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                    `).slideDown();
                } else if (multiscaleMethod === 'maxmatch') {
                    thisObj.multiscaleParamsDiv.html(`
                        <div class="form-group row p-1">
                            <label for="th-range" class="col-sm-4 col-form-label col-form-label-sm">threshold</label>
                            <input type="range" name="th" class="col-sm-4 custom-range mt-2" min="20" max="200" step="1" id="th-range"
                                   value="100"/>
                            <input type="text" class="col-sm-3 offset-sm-1 border-0"/>
                        </div>
                    `).slideDown();
                }
                else if (multiscaleMethod === 'none'){
                    thisObj.multiscaleParamsDiv.html('');
                }

                thisObj.multiscaleParamsDiv.find('input[type=range]').mousemove(function () {
                    $(this).siblings('input[type=text]').val(this.value);
                }).mousemove();
                thisObj.multiscaleParamsDiv.find('input[type=text]').change(function () {
                    $(this).siblings('input[type=range]').val(parseFloat(this.value));
                });
            });
        }
    }

    fnGetParams(){
        let params = {};
        this.multiscaleParamsDiv.find('input[type=range]').forEach(function(){
            params[this.name] = this.value;
        });
        return params;
    }
}


class LeftDiv {
    constructor(observer, requestDatasetInfoUrl) {
        this.name = 'LeftDiv';
        this.observer = observer;
        observer.fnAddView(this);
        this.$datasetSelect = $('#dataset-select');
        this.$layoutSelect = $('#layout-select');
        this.layoutParams = new LayoutParamsDiv();
        this.$multiscaleSelect = $('#multiscale-select');
        this.multiscaleParams = new MultiscaleParamsDiv();

        this.datasetInfoDiv = new DatasetInfoDiv();

        this.requestDatasetInfoUrl = requestDatasetInfoUrl;
    }

    fnInitialize() {
        let thisObj = this;

        // dataset select
        thisObj.$datasetSelect.change(function () {
            let selected_dataset = $(this).val();
            console.log('selected_dataset', selected_dataset);
            if (selected_dataset === 'none') return;

            $.ajax({
                url: thisObj.requestDatasetInfoUrl,
                data: {datasetname: selected_dataset},
                dataType: 'json',
                method: 'POST',
                success: function (datasetInfo) {  // datasetInfo: {success: S, data: dataset_info}
                    if (datasetInfo.hasOwnProperty('success') && datasetInfo.success) {
                        thisObj.datasetInfoDiv.fnUpdateTable(datasetInfo['data']);
                    } else {
                        thisObj.datasetInfoDiv.fnUpdateError();
                    }
                },
                error: function () {
                    thisObj.datasetInfoDiv.fnUpdateError();
                }
            })
        });

        // layout select
        thisObj.$layoutSelect.change(function () {
            let layoutMethod = this.value;

            thisObj.layoutParams.fnUpdate(layoutMethod);
        });

        // multiscale select
        thisObj.$multiscaleSelect.change(function () {
            let multiscaleMethod = this.value;

            thisObj.multiscaleParams.fnUpdate(multiscaleMethod);
        });

        // start button
        $('#start-drawing-button').click(function(){
            // 检查数据是否合理
            if (thisObj.$layoutSelect.val() === 'none'){
                // TODO: 处理输入参数不合理情形
            }
            else{
                let params = {datasetname: null, layout: {}, multiscale: {}};
                params.datasetname = thisObj.$datasetSelect.val();
                params.layout.method = thisObj.$layoutSelect.val();
                params.multiscale.method = thisObj.$multiscaleSelect.val();
                // layout参数输入
                let layoutParams = thisObj.layoutParams.fnGetParams();
                for (let k in layoutParams){
                    if (layoutParams.hasOwnProperty(k)){
                        params.layout[k] = layoutParams[k];
                    }
                }
                let multiscaleParams = thisObj.multiscaleParams.fnGetParams();
                for (let k in multiscaleParams){
                    if (multiscaleParams.hasOwnProperty(k)){
                        params.multiscale[k] = multiscaleParams[k];
                    }
                }
                thisObj.observer.fnFireEvent('start_draw', params, thisObj.name);
            }
        });

        // stop button
        $('#stop-drawing-button').click(function(){
            thisObj.observer.fnFireEvent('stop_draw', null, thisObj.name);
        });
    }

    fnOnMessage(message, data, from) {

    }
}