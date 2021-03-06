(function () {
  'use strict';
  var app = angular.module("CovidApp", [])
    .directive("ngUploadChange",function(){
      return{
        scope:{
          ngUploadChange:"&"
        },
        link:function($scope, $element, $attrs){
          $element.on("change",function(event){
            $scope.$apply(function(){
              $scope.ngUploadChange({$event: event})
            })
          })
          $scope.$on("$destroy",function(){
            $element.off();
          });
        }
      }
    });

  app.run(function($rootScope) {
    $rootScope.onKeyDown = function(e) {
      if (e.keyCode === 37 || e.keyCode === 32) {
        $rootScope.$broadcast("arrowLeft");
        e.preventDefault();  
      } else if (e.keyCode === 38) {
        $rootScope.$broadcast("arrowUp");
        e.preventDefault();  
      } else if (e.keyCode === 39) {
        $rootScope.$broadcast("arrowRight");
        e.preventDefault();  
      } else if (e.keyCode === 40) {
        $rootScope.$broadcast("arrowDown");
        e.preventDefault();  
      }
    };
  });

  app.controller('CovidController', ['$scope', '$log', '$http', '$timeout',
    function($scope, $log, $http, $timeout) {
      var vm = this;
      $scope.submitButtonText = 'Submit';
      $scope.loading = false;
      $scope.error = false;

      if (jobKey) {
        $scope.loading = true;
        pollForResults(jobKey);
      } else {
        assignData(data);
      }

      function assignData(data) {
        $scope.currentSlice = 0;
        $scope.currentOverlay = 1;
        $scope.ctSlices = data.origUrls;
        $scope.predSlices = data.predUrls;
        $scope.refSlices = data.refUrls;
        vm.numSlices = $scope.ctSlices.length;
        vm.numOverlays = ($scope.refSlices.length > 0 ? 3 : 2);
        $scope.hasRef = (vm.numOverlays == 3);
      }

      function pollForResults(jobKey) {
        var timeout = "";

        $log.log("Polling");
        var poller = function() {
          $http.post('/results/' + jobKey, {}).
            success(function(data, status, headers, config) {
              if (status === 202) {
                $log.log(data, status);
                $scope.error = data.statusText;
              } else if (status === 200){
                $scope.loading = false;
                $log.log(data);
                assignData(data);
                $timeout.cancel(timeout);
                return false;
              }
              timeout = $timeout(poller, 2000);
            });
        };
        poller();
      }

      $scope.changeSlice = function(delta) {
        $scope.currentSlice = ($scope.currentSlice + vm.numSlices + delta) % vm.numSlices;
        $scope.currentOverlay = 1;
      }
      $scope.changeOverlay = function(delta) {
        $scope.currentOverlay = ($scope.currentOverlay + vm.numOverlays + delta) % vm.numOverlays;
      }

      $scope.$on("arrowUp", function() {
        $scope.changeSlice(1);
      });
      $scope.$on("arrowDown", function() {
        $scope.changeSlice(-1);
      });
      $scope.$on("arrowLeft", function() {
        $scope.changeOverlay(-1);
      });
      $scope.$on("arrowRight", function() {
        $scope.changeOverlay(1);
      });

      $scope.onUploadChange = function($event){
        var files = $event.target.files;

        let formData = new FormData();
        formData.append('files', files);
        return $http.post('/upload', formData).then(function successCallback(response) {
          $log.log(response);
          $scope.error = false;
        }, function errorCallback(response) {
          $scope.error = response.statusText;
        });
      }
    }
  ]);
}());
