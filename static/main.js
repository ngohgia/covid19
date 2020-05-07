(function () {
  'use strict';
  var app = angular.module("CovidApp", []);
  app.run(function($rootScope) {
    $rootScope.onKeyDown = function(e) {
      if (e.keyCode === 37) {
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

      // $log.log(data);

      $scope.currentSlice = 0;
      $scope.currentOverlay = 1;
      $scope.ctSlices = data.origUrls;
      $scope.predSlices = data.predUrls;
      $scope.refSlices = data.refUrls;
      vm.numSlices = $scope.ctSlices.length;
      vm.numOverlays = ($scope.refSlices.length > 0 ? 3 : 2);


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
    }
  ]);
}());
