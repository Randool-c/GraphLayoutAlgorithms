class Observer{
    constructor() {
        this.views = [];
    }

    fnAddView(newView){
        this.views.push(newView);
    }

    fnFireEvent(message, data, from){
        this.views.forEach(function(view){
            view.fnOnMessage(message, data, from);
        })
    }
}