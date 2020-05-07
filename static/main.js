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

      vm.numSlices = 3;
      vm.numOverlays = 3;
      $scope.currentSlice = 0;
      $scope.currentOverlay = 1;
      $scope.ctSlices= [
        '/eval/orig/0.jpg',
        '/eval/orig/1.jpg',
        '/eval/orig/2.jpg',
      ];

      $scope.predSlices= [
        '/eval/pred/0.png',
        '/eval/pred/1.png',
        '/eval/pred/2.png',
      ];

      $scope.refSlices= [
        '/eval/ref/0.png',
        '/eval/ref/1.png',
        '/eval/ref/2.png',
      ];

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
